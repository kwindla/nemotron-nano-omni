import asyncio
import contextlib
import copy
import sys
import unittest
from pathlib import Path
from typing import Any

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import (
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage, NOT_GIVEN
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_voice.bot import _build_llm_context  # noqa: E402
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    ChatCompletionPassResult,
    DEFAULT_VOICE_SYSTEM_INSTRUCTION,
    NemotronOmniAudioLLMService,
)


class _DummySession:
    async def close(self) -> None:
        return None


class NemotronOmniAlignedTests(unittest.IsolatedAsyncioTestCase):
    def _make_service(
        self,
        *,
        enable_bash_tool: bool = True,
        system_instruction: str | None = DEFAULT_VOICE_SYSTEM_INSTRUCTION,
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(enable_bash_tool=enable_bash_tool)
        service._settings.system_instruction = system_instruction

        async def noop(*args, **kwargs):
            return None

        async def cancel_task(task, timeout=None):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        service.start_processing_metrics = noop  # type: ignore[method-assign]
        service.stop_processing_metrics = noop  # type: ignore[method-assign]
        service.start_ttfb_metrics = noop  # type: ignore[method-assign]
        service.stop_ttfb_metrics = noop  # type: ignore[method-assign]
        service.start_llm_usage_metrics = noop  # type: ignore[method-assign]
        service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)  # type: ignore[method-assign]
        service.cancel_task = cancel_task  # type: ignore[method-assign]
        service._session = _DummySession()
        return service

    def _attach_assistant(
        self,
        service: NemotronOmniAudioLLMService,
        context: LLMContext,
    ) -> tuple[LLMAssistantAggregator, list[Any], list[Any]]:
        assistant = LLMAssistantAggregator(context)
        service_frames: list[Any] = []
        assistant_frames: list[Any] = []

        async def capture_assistant_frame(
            frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
        ):
            assistant_frames.append(frame)

        async def forward_service_frame(
            frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
        ):
            service_frames.append(frame)
            await assistant.process_frame(frame, direction)

        assistant.push_frame = capture_assistant_frame  # type: ignore[method-assign]
        service.push_frame = forward_service_frame  # type: ignore[method-assign]
        return assistant, service_frames, assistant_frames

    def _text_stream(
        self,
        service: NemotronOmniAudioLLMService,
        payloads: list[dict[str, Any]],
        *,
        text: str,
        chunks: list[str] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        first_token: bool = False,
    ):
        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            for chunk in chunks if chunks is not None else ([text] if text else []):
                if chunk:
                    await service._push_llm_text(chunk)
            return ChatCompletionPassResult(
                output_text=text,
                tool_calls=tool_calls or [],
                first_token=first_token,
            )

        return fake_stream

    async def _run_context_frame(
        self,
        service: NemotronOmniAudioLLMService,
        context: LLMContext,
    ) -> None:
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        task = service._generation_task
        if task is not None:
            await task

    def test_openai_tool_definition_shape_matches_expected_bash_tool_definition(self) -> None:
        definition = BASH_TOOL_DEFINITION["function"]
        parameters = definition["parameters"]
        description = definition["description"]

        self.assertIs(parameters["additionalProperties"], False)
        self.assertIn("JSON observation object", description)
        for field_name in (
            "ok",
            "status",
            "summary",
            "command",
            "exit_code",
            "timed_out",
            "stdout",
            "stderr",
        ):
            self.assertIn(field_name, description)
        self.assertIn("duplicate_suppressed", description)
        self.assertIn("round_limit_reached", description)

    def test_bot_builds_tools_schema_context_only_when_bash_tool_enabled(self) -> None:
        enabled_context = _build_llm_context(enable_bash_tool=True)
        disabled_context = _build_llm_context(enable_bash_tool=False)

        self.assertIsInstance(enabled_context.tools, ToolsSchema)
        self.assertEqual(enabled_context.tool_choice, "auto")
        self.assertEqual(enabled_context.tools.standard_tools, [])
        self.assertIn(AdapterType.OPENAI, enabled_context.tools.custom_tools or {})
        custom_tools = enabled_context.tools.custom_tools or {}
        self.assertEqual(custom_tools[AdapterType.OPENAI][0], BASH_TOOL_DEFINITION)
        self.assertIsNot(custom_tools[AdapterType.OPENAI][0], BASH_TOOL_DEFINITION)

        self.assertIs(disabled_context.tools, NOT_GIVEN)
        self.assertIs(disabled_context.tool_choice, NOT_GIVEN)

    async def test_response_lifecycle_frames_pair_under_supersession(self) -> None:
        service = self._make_service()
        seen_frames: list[str] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def capture_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            if isinstance(frame, LLMFullResponseEndFrame):
                await asyncio.sleep(0)
            seen_frames.append(type(frame).__name__)

        async def fake_process(context: LLMContext):
            user_text = context.get_messages()[-1]["content"]
            if user_text == "first":
                first_started.set()
                await release_first.wait()
            else:
                return None

        service.push_frame = capture_frame  # type: ignore[method-assign]
        service._process_context = fake_process  # type: ignore[method-assign]

        first_context = LLMContext(messages=[{"role": "user", "content": "first"}])
        second_context = LLMContext(messages=[{"role": "user", "content": "second"}])

        await service.process_frame(LLMContextFrame(first_context), FrameDirection.DOWNSTREAM)
        await first_started.wait()
        await service.process_frame(LLMContextFrame(second_context), FrameDirection.DOWNSTREAM)
        release_first.set()
        if service._generation_task is not None:
            await service._generation_task

        self.assertEqual(
            seen_frames,
            [
                "LLMFullResponseStartFrame",
                "LLMFullResponseEndFrame",
                "LLMFullResponseStartFrame",
                "LLMFullResponseEndFrame",
            ],
        )

    async def test_developer_messages_follow_openai_adapter_policy_and_downgrade_to_user_not_system(
        self,
    ) -> None:
        context = LLMContext(
            messages=[
                {"role": "developer", "content": "Developer instructions stay user-visible."},
                {"role": "user", "content": "What should I do next?"},
            ]
        )
        service = self._make_service(system_instruction="service system")
        _, _, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Answer",
        )

        await self._run_context_frame(service, context)

        developer_rows = [
            message
            for message in payloads[0]["messages"]
            if message.get("content") == "Developer instructions stay user-visible."
        ]
        self.assertEqual(developer_rows, [{"role": "user", "content": developer_rows[0]["content"]}])
        self.assertNotIn(
            {
                "role": "system",
                "content": "Developer instructions stay user-visible.",
            },
            payloads[0]["messages"],
        )

    async def test_system_instruction_injection_stays_separate_from_developer_message_normalization(
        self,
    ) -> None:
        context = LLMContext(
            messages=[
                {"role": "developer", "content": "Developer policy."},
                {"role": "user", "content": "Question"},
            ]
        )
        service = self._make_service(system_instruction="service-level system instruction")
        _, _, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Answer",
        )

        await self._run_context_frame(service, context)

        self.assertEqual(
            payloads[0]["messages"][0],
            {"role": "system", "content": "service-level system instruction"},
        )
        self.assertIn(
            {"role": "user", "content": "Developer policy."},
            payloads[0]["messages"],
        )

    async def test_matching_llm_specific_messages_are_unwrapped_and_nonmatching_ones_are_excluded(
        self,
    ) -> None:
        anthropic_message = LLMSpecificMessage(
            llm="anthropic",
            message={"role": "assistant", "content": "Anthropic-only prompt row."},
        )
        openai_message = LLMSpecificMessage(
            llm="openai",
            message={"role": "assistant", "content": "OpenAI-only prompt row."},
        )
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Standard user row."},
                anthropic_message,
                openai_message,
                {"role": "user", "content": "Final user row."},
            ]
        )
        service = self._make_service(system_instruction="service system")
        _, _, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []

        def capture_trace(*, trace_id: str, phase: str, payload: dict[str, Any]) -> None:
            traces.append({"trace_id": trace_id, "phase": phase, "payload": copy.deepcopy(payload)})

        service._write_trace_file = capture_trace  # type: ignore[method-assign]
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Answer",
        )

        await self._run_context_frame(service, context)

        prompt_messages = payloads[0]["messages"]
        self.assertIn(
            {"role": "assistant", "content": "OpenAI-only prompt row."},
            prompt_messages,
        )
        self.assertNotIn(
            {"role": "assistant", "content": "Anthropic-only prompt row."},
            prompt_messages,
        )
        self.assertIn(anthropic_message, context.get_messages())
        self.assertIn(openai_message, context.get_messages())

        request_traces = [trace for trace in traces if trace["phase"] == "client-request"]
        self.assertEqual(len(request_traces), 1)
        self.assertEqual(
            request_traces[0]["payload"]["conversation_full_messages"],
            service._context_lineage_messages,
        )
        self.assertEqual(request_traces[0]["payload"]["conversation_full_messages"], prompt_messages)
        self.assertNotIn(
            {"role": "assistant", "content": "Anthropic-only prompt row."},
            request_traces[0]["payload"]["conversation_full_messages"],
        )

    async def test_tools_and_tool_choice_are_omitted_when_provider_tools_empty(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "No tools this turn."}])
        service = self._make_service(enable_bash_tool=False, system_instruction="service system")
        _, _, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Answer",
        )

        await self._run_context_frame(service, context)

        self.assertNotIn("tools", payloads[0])
        self.assertNotIn("tool_choice", payloads[0])

    async def test_one_turn_text_response(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        service = self._make_service()
        _, service_frames, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Hello there",
        )

        await self._run_context_frame(service, context)

        self.assertEqual(
            context.get_messages(),
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello there"},
            ],
        )
        self.assertEqual(
            [type(frame) for frame in service_frames],
            [LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
        )

    async def test_shared_context_text_response_matches_text_only_stream(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "Say hello"}])
        service = self._make_service()
        _, service_frames, _ = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Hello there",
            chunks=["Hello ", "there"],
        )

        await self._run_context_frame(service, context)

        self.assertEqual(context.get_messages()[-1]["content"], "Hello there")
        self.assertEqual(
            [frame.text for frame in service_frames if isinstance(frame, LLMTextFrame)],
            ["Hello ", "there"],
        )

    async def test_text_only_path_preserves_stock_assistant_aggregator_behavior(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "Summarize."}])
        service = self._make_service()
        _, _, assistant_frames = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="Short answer",
            chunks=["Short ", "answer"],
        )

        await self._run_context_frame(service, context)

        self.assertEqual(
            [type(frame) for frame in assistant_frames],
            [LLMContextFrame, LLMContextAssistantTimestampFrame],
        )
        self.assertEqual(context.get_messages()[-1], {"role": "assistant", "content": "Short answer"})

    async def test_empty_pass_does_not_append_empty_assistant_row(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "Stay silent."}])
        service = self._make_service()
        _, _, assistant_frames = self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        service._stream_completion_pass = self._text_stream(  # type: ignore[method-assign]
            service,
            payloads,
            text="",
            chunks=[],
            first_token=True,
        )

        await self._run_context_frame(service, context)

        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Stay silent."}])
        self.assertEqual(assistant_frames, [])


if __name__ == "__main__":
    unittest.main()
