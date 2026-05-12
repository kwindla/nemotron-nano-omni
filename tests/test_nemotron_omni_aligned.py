import asyncio
import contextlib
import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Callable

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.base_task import PipelineTaskParams
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage, NOT_GIVEN
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_voice.bot import (  # noqa: E402
    AudioOnlyLLMUserAggregator,
    UserAudioContextCollector,
    _build_llm_context,
)
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    ChatCompletionPassResult,
    DEFAULT_VOICE_SYSTEM_INSTRUCTION,
    NemotronOmniAudioLLMService,
)


class _DummySession:
    async def close(self) -> None:
        return None


class _RecordingPassthroughProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.seen_frames: list[tuple[Any, FrameDirection]] = []

    async def process_frame(self, frame, direction: FrameDirection):
        self.seen_frames.append((frame, direction))
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _RecordingUserAggregator(AudioOnlyLLMUserAggregator):
    def __init__(self, context: LLMContext):
        super().__init__(context)
        self.seen_frames: list[tuple[Any, FrameDirection]] = []

    async def process_frame(self, frame, direction: FrameDirection):
        self.seen_frames.append((frame, direction))
        await super().process_frame(frame, direction)


class _RecordingAudioCollector(UserAudioContextCollector):
    def __init__(self, *, context: LLMContext, user_aggregator: AudioOnlyLLMUserAggregator):
        super().__init__(
            context=context,
            user_aggregator=user_aggregator,
            audio_context_text="User audio follows.",
            push_context_on_finish=False,
        )
        self.seen_frames: list[tuple[Any, FrameDirection]] = []

    async def process_frame(self, frame, direction: FrameDirection):
        self.seen_frames.append((frame, direction))
        await super().process_frame(frame, direction)


class _RecordingOutputTransport(BaseOutputTransport):
    def __init__(self):
        super().__init__(
            TransportParams(
                audio_out_enabled=False,
                camera_out_enabled=False,
                audio_out_sample_rate=24000,
                audio_out_channels=1,
            )
        )
        self.seen_frames: list[tuple[Any, FrameDirection]] = []

    async def process_frame(self, frame, direction: FrameDirection):
        self.seen_frames.append((frame, direction))
        await FrameProcessor.process_frame(self, frame, direction)
        await self.push_frame(frame, direction)

    async def send_message(self, frame):
        return None

    async def register_video_destination(self, destination: str):
        return None

    async def register_audio_destination(self, destination: str):
        return None


class _DummyBroadcastLLM(FrameProcessor):
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self.followup_frame_ids: list[int] = []
        self.broadcasted = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction is FrameDirection.DOWNSTREAM:
            if self.broadcasted:
                return
            self.broadcasted = True
            function_call = FunctionCallFromLLM(
                function_name="run_bash",
                tool_call_id="call_route_1",
                arguments={"code": "pwd"},
                context=self._context,
            )
            await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=[function_call])
            await self.broadcast_frame(
                FunctionCallInProgressFrame,
                function_name="run_bash",
                tool_call_id="call_route_1",
                arguments={"code": "pwd"},
                cancel_on_interruption=True,
                group_id=None,
            )
            await self.broadcast_frame(
                FunctionCallResultFrame,
                function_name="run_bash",
                tool_call_id="call_route_1",
                arguments={"code": "pwd"},
                result={"ok": True, "status": "success"},
                run_llm=True,
            )
            return
        if isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM:
            self.followup_frame_ids.append(frame.id)
            return
        await self.push_frame(frame, direction)


class _UpstreamContextConsumer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.seen_upstream_ids: list[int] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM:
            self.seen_upstream_ids.append(frame.id)
            return
        await self.push_frame(frame, direction)


class _DummySerialToolRoundLLM(FrameProcessor):
    def __init__(self, context: LLMContext, commands: list[str]):
        super().__init__()
        self._context = context
        self._commands = commands
        self.initial_requests = 0
        self.followup_requests = 0
        self.executed_commands: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction is FrameDirection.DOWNSTREAM:
            self.initial_requests += 1
            function_calls = []
            for index, command in enumerate(self._commands, start=1):
                function_calls.append(
                    FunctionCallFromLLM(
                        function_name="run_bash",
                        tool_call_id=f"call_{index}",
                        arguments={"code": command},
                        context=self._context,
                    )
                )
            await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=function_calls)
            for index, function_call in enumerate(function_calls):
                self.executed_commands.append(function_call.arguments["code"])
                await self.broadcast_frame(
                    FunctionCallInProgressFrame,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    cancel_on_interruption=True,
                    group_id=None,
                )
                await self.broadcast_frame(
                    FunctionCallResultFrame,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    result={"ok": True, "status": "success"},
                    run_llm=index == len(function_calls) - 1,
                )
            return
        if isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM:
            self.followup_requests += 1
            return
        await self.push_frame(frame, direction)


class _DummyToolFollowupAssistant(FrameProcessor):
    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context
        self.followup_pushes = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, FunctionCallResultFrame)
            and direction is FrameDirection.DOWNSTREAM
            and frame.run_llm
        ):
            self.followup_pushes += 1
            await self.push_frame(LLMContextFrame(self._context), FrameDirection.UPSTREAM)
            return
        await self.push_frame(frame, direction)


class NemotronOmniAlignedTests(unittest.IsolatedAsyncioTestCase):
    def _make_service(
        self,
        *,
        enable_bash_tool: bool = True,
        system_instruction: str | None = DEFAULT_VOICE_SYSTEM_INSTRUCTION,
        bash_tool_max_rounds: int = 3,
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(
            enable_bash_tool=enable_bash_tool,
            bash_tool_max_rounds=bash_tool_max_rounds,
        )
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

    def _context_with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_bash_tool: bool = True,
    ) -> LLMContext:
        template = _build_llm_context(enable_bash_tool=enable_bash_tool)
        return LLMContext(
            messages=copy.deepcopy(messages),
            tools=template.tools,
            tool_choice=template.tool_choice,
        )

    async def _prime_service_for_tools(self, service: NemotronOmniAudioLLMService) -> None:
        await service._create_sequential_runner_task()
        self.addAsyncCleanup(service._cancel_sequential_runner_task)

    async def _wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        timeout: float = 1.0,
    ) -> None:
        async def _poll():
            while not predicate():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_poll(), timeout)

    def _attach_assistant(
        self,
        service: NemotronOmniAudioLLMService,
        context: LLMContext,
        *,
        auto_reenter: bool = False,
        on_assistant_push: Callable[[Any, FrameDirection], Any] | None = None,
    ) -> tuple[LLMAssistantAggregator, list[tuple[Any, FrameDirection]], list[tuple[Any, FrameDirection]]]:
        assistant = LLMAssistantAggregator(context)
        assistant.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)  # type: ignore[method-assign]

        async def cancel_task(task, timeout=None):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assistant.cancel_task = cancel_task  # type: ignore[method-assign]
        service_frames: list[tuple[Any, FrameDirection]] = []
        assistant_frames: list[tuple[Any, FrameDirection]] = []

        async def capture_assistant_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            assistant_frames.append((frame, direction))
            if on_assistant_push is not None:
                maybe_awaitable = on_assistant_push(frame, direction)
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
            if auto_reenter and isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM:
                await service.process_frame(frame, direction)

        async def forward_service_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            service_frames.append((frame, direction))
            if direction is FrameDirection.DOWNSTREAM:
                await assistant.process_frame(frame, direction)

        assistant.push_frame = capture_assistant_frame  # type: ignore[method-assign]
        service.push_frame = forward_service_frame  # type: ignore[method-assign]
        return assistant, service_frames, assistant_frames

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
        await self._prime_service_for_tools(service)
        seen_frames: list[str] = []
        first_tool_started = asyncio.Event()
        release_first_tool = asyncio.Event()

        context = self._context_with_messages([{"role": "user", "content": "first"}])

        async def capture_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            if direction is FrameDirection.DOWNSTREAM:
                seen_frames.append(type(frame).__name__)

        async def fake_stream(payload, **kwargs):
            last_user = payload["messages"][-1]["content"]
            if last_user == "first":
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"sleep"}'},
                        }
                    ],
                    first_token=False,
                )
            return ChatCompletionPassResult(output_text="second done", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            first_tool_started.set()
            await release_first_tool.wait()
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "",
                "stderr": "",
            }

        service.push_frame = capture_frame  # type: ignore[method-assign]
        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]

        first_task = asyncio.create_task(
            service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        )
        await first_tool_started.wait()
        second_context = self._context_with_messages([{"role": "user", "content": "second"}])
        await service.process_frame(LLMContextFrame(second_context), FrameDirection.DOWNSTREAM)
        release_first_tool.set()
        await first_task
        if service._generation_task is not None:
            await service._generation_task

        lifecycle_frames = [
            frame_name
            for frame_name in seen_frames
            if frame_name in {"LLMFullResponseStartFrame", "LLMFullResponseEndFrame"}
        ]
        self.assertEqual(
            lifecycle_frames,
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
        context = self._context_with_messages(
            [
                {"role": "developer", "content": "Developer instructions stay user-visible."},
                {"role": "user", "content": "What should I do next?"},
            ]
        )
        service = self._make_service(system_instruction="service system")
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            await service._push_llm_text("Answer")
            return ChatCompletionPassResult(output_text="Answer", tool_calls=[], first_token=False)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        developer_rows = [
            message
            for message in payloads[0]["messages"]
            if message.get("content") == "Developer instructions stay user-visible."
        ]
        self.assertEqual(
            developer_rows,
            [{"role": "user", "content": "Developer instructions stay user-visible."}],
        )
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
        context = self._context_with_messages(
            [
                {"role": "developer", "content": "Developer policy."},
                {"role": "user", "content": "Question"},
            ]
        )
        service = self._make_service(system_instruction="service-level system instruction")
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            await service._push_llm_text("Answer")
            return ChatCompletionPassResult(output_text="Answer", tool_calls=[], first_token=False)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
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
        context = self._context_with_messages(
            [
                {"role": "user", "content": "Standard user row."},
                anthropic_message,
                openai_message,
                {"role": "user", "content": "Final user row."},
            ]
        )
        service = self._make_service(system_instruction="service system")
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []

        def capture_trace(*, trace_id: str, phase: str, payload: dict[str, Any]) -> None:
            traces.append({"trace_id": trace_id, "phase": phase, "payload": copy.deepcopy(payload)})

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            await service._push_llm_text("Answer")
            return ChatCompletionPassResult(output_text="Answer", tool_calls=[], first_token=False)

        service._write_trace_file = capture_trace  # type: ignore[method-assign]
        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
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
            prompt_messages,
        )

    async def test_tools_and_tool_choice_are_omitted_when_provider_tools_empty(self) -> None:
        context = LLMContext(messages=[{"role": "user", "content": "No tools this turn."}])
        service = self._make_service(enable_bash_tool=False, system_instruction="service system")
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            await service._push_llm_text("Answer")
            return ChatCompletionPassResult(output_text="Answer", tool_calls=[], first_token=False)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        self.assertNotIn("tools", payloads[0])
        self.assertNotIn("tool_choice", payloads[0])

    async def test_one_turn_text_response(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "Hello"}])
        service = self._make_service()
        _, service_frames, _ = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Hello there")
            return ChatCompletionPassResult(
                output_text="Hello there",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        self.assertEqual(
            context.get_messages(),
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello there"},
            ],
        )
        self.assertEqual(
            [type(frame) for frame, direction in service_frames if direction is FrameDirection.DOWNSTREAM],
            [LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
        )

    async def test_shared_context_text_response_matches_text_only_stream(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "Say hello"}])
        service = self._make_service()
        _, service_frames, _ = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Hello ")
            await service._push_llm_text("there")
            return ChatCompletionPassResult(output_text="Hello there", tool_calls=[], first_token=False)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        self.assertEqual(context.get_messages()[-1]["content"], "Hello there")
        self.assertEqual(
            [frame.text for frame, direction in service_frames if direction is FrameDirection.DOWNSTREAM and isinstance(frame, LLMTextFrame)],
            ["Hello ", "there"],
        )

    async def test_text_only_path_preserves_stock_assistant_aggregator_behavior(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "Summarize."}])
        service = self._make_service()
        _, _, assistant_frames = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Short ")
            await service._push_llm_text("answer")
            return ChatCompletionPassResult(output_text="Short answer", tool_calls=[], first_token=False)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        self.assertEqual(
            [type(frame) for frame, _ in assistant_frames],
            [LLMContextFrame, LLMContextAssistantTimestampFrame],
        )
        self.assertEqual(context.get_messages()[-1], {"role": "assistant", "content": "Short answer"})

    async def test_empty_pass_does_not_append_empty_assistant_row(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "Stay silent."}])
        service = self._make_service()
        _, _, assistant_frames = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            return ChatCompletionPassResult(output_text="", tool_calls=[], first_token=True)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)

        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Stay silent."}])
        self.assertEqual(assistant_frames, [])

    async def test_multi_tool_round_executes_serially_and_reenters_after_final_result(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Inspect the workspace."}])
        _, service_frames, _ = self._attach_assistant(service, context, auto_reenter=True)

        payloads: list[dict[str, Any]] = []
        execution_order: list[str] = []
        followup_order_snapshot: list[str] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"ls"}'},
                        },
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"whoami"}'},
                        },
                    ],
                    first_token=False,
                )
            followup_order_snapshot[:] = execution_order
            await service._push_llm_text("done")
            return ChatCompletionPassResult(output_text="done", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            execution_order.append(code)
            await asyncio.sleep(0)
            return {
                "ok": True,
                "status": "success",
                "summary": f"ran {code}",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)

        await self._wait_until(
            lambda: len(payloads) == 2
            and context.get_messages()[-1] == {"role": "assistant", "content": "done"}
        )

        tool_result_frames = [
            frame
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallResultFrame)
        ]
        self.assertEqual(execution_order, ["pwd", "ls", "whoami"])
        self.assertEqual(followup_order_snapshot, ["pwd", "ls", "whoami"])
        self.assertEqual(len(payloads), 2)
        self.assertEqual([frame.run_llm for frame in tool_result_frames], [False, False, True])

    async def test_sync_tool_round_round_trip(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Run pwd and explain it."}])
        self._attach_assistant(service, context, auto_reenter=True)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("The working directory is the repo root.")
            return ChatCompletionPassResult(
                output_text="The working directory is the repo root.",
                tool_calls=[],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            return {
                "ok": True,
                "status": "success",
                "summary": "Command completed successfully.",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 2
            and context.get_messages()[-1]["content"] == "The working directory is the repo root."
        )

        tool_rows = [message for message in context.get_messages() if message.get("role") == "tool"]
        self.assertEqual(len(tool_rows), 1)
        self.assertEqual(json.loads(tool_rows[0]["content"])["stdout"], "/repo\n")
        self.assertEqual(len(payloads), 2)

    async def test_interrupted_multi_tool_batch_gives_every_tool_call_id_a_terminal_lifecycle_signal(
        self,
    ) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Run several commands."}])
        _, service_frames, _ = self._attach_assistant(service, context)
        started = asyncio.Event()
        cancelled = asyncio.Event()
        cancelled_batches: list[list[str]] = []

        async def capture_event(name: str, *args, **kwargs):
            if name == "on_function_calls_cancelled":
                cancelled_batches.append([call.tool_call_id for call in args[0]])
            return None

        async def fake_stream(payload, **kwargs):
            return ChatCompletionPassResult(
                output_text="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"sleep"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"ls"}'},
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                    },
                ],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        service._call_event_handler = capture_event  # type: ignore[method-assign]
        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]

        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await cancelled.wait()
        await self._wait_until(
            lambda: sum(
                1
                for frame, direction in service_frames
                if direction is FrameDirection.DOWNSTREAM
                and isinstance(frame, FunctionCallCancelFrame)
            )
            == 3
        )

        cancel_ids = [
            frame.tool_call_id
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallCancelFrame)
        ]
        result_ids = [
            frame.tool_call_id
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallResultFrame)
        ]
        self.assertEqual(set(cancel_ids), {"call_1", "call_2", "call_3"})
        self.assertEqual(result_ids, [])
        self.assertEqual(
            {tool_call_id for batch in cancelled_batches for tool_call_id in batch},
            {"call_1", "call_2", "call_3"},
        )

    async def test_stale_queued_sync_tool_cancellation_fires_on_function_calls_cancelled(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Cancel queued work."}])
        started = asyncio.Event()
        queued_cancel_batch: list[str] = []

        async def capture_event(name: str, *args, **kwargs):
            nonlocal queued_cancel_batch
            if name == "on_function_calls_cancelled":
                batch_ids = [call.tool_call_id for call in args[0]]
                if len(batch_ids) > 1:
                    queued_cancel_batch = batch_ids
            return None

        async def fake_stream(payload, **kwargs):
            return ChatCompletionPassResult(
                output_text="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"sleep"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"ls"}'},
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                    },
                ],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            started.set()
            await asyncio.Future()

        service._call_event_handler = capture_event  # type: ignore[method-assign]
        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(lambda: queued_cancel_batch == ["call_2", "call_3"])

    async def test_cancel_running_sync_tool_does_not_kill_sequential_runner(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        first_context = self._context_with_messages([{"role": "user", "content": "First"}])
        second_context = self._context_with_messages([{"role": "user", "content": "Second"}])
        self._attach_assistant(service, second_context, auto_reenter=True)

        payloads: list[dict[str, Any]] = []
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        executed_codes: list[str] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            last_user = payload["messages"][-1]["content"]
            if last_user == "First":
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_sleep",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"sleep"}'},
                        }
                    ],
                    first_token=False,
                )
            if len(payloads) == 2:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_pwd",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("runner survived")
            return ChatCompletionPassResult(output_text="runner survived", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            if code == "sleep":
                first_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            executed_codes.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(first_context), FrameDirection.DOWNSTREAM)
        await first_started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await first_cancelled.wait()
        self.assertFalse(service._sequential_runner_task.done())

        self._attach_assistant(service, second_context, auto_reenter=True)
        await service.process_frame(LLMContextFrame(second_context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: second_context.get_messages()
            and second_context.get_messages()[-1] == {"role": "assistant", "content": "runner survived"}
        )
        self.assertEqual(executed_codes, ["pwd"])

    async def test_new_user_turn_without_interruption_supersedes_in_flight_tool_batch(self) -> None:
        # A new top-level user turn arriving as a bare LLMContextFrame (no
        # preceding InterruptionFrame) must still cancel the in-flight tool,
        # not leak the old batch's per-turn state into the new turn, and not
        # let a stale run_llm=True followup from the old batch supersede the
        # new turn.
        service = self._make_service()
        await self._prime_service_for_tools(service)
        first_context = self._context_with_messages([{"role": "user", "content": "First turn"}])
        second_context = self._context_with_messages([{"role": "user", "content": "Second turn"}])
        payloads: list[dict[str, Any]] = []
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        executed_codes: list[str] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_old",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"sleep"}'},
                        }
                    ],
                    first_token=False,
                )
            if len(payloads) == 2:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_new",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("second turn answer")
            return ChatCompletionPassResult(output_text="second turn answer", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            if code == "sleep":
                first_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            executed_codes.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        _, service_frames, _ = self._attach_assistant(service, second_context, auto_reenter=True)
        await service.process_frame(LLMContextFrame(first_context), FrameDirection.DOWNSTREAM)
        await first_started.wait()
        # New turn, NO InterruptionFrame.
        await service.process_frame(LLMContextFrame(second_context), FrameDirection.DOWNSTREAM)
        await first_cancelled.wait()
        await self._wait_until(
            lambda: second_context.get_messages()
            and second_context.get_messages()[-1] == {"role": "assistant", "content": "second turn answer"}
        )

        self.assertFalse(service._sequential_runner_task.done())
        self.assertEqual(executed_codes, ["pwd"])
        # The superseded "sleep" tool was cancelled, not completed — it must not
        # have leaked dedup state into the new turn nor emitted a result frame.
        self.assertNotIn(("run_bash", "sleep"), service._turn_tool_results)
        result_ids = [
            frame.tool_call_id
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallResultFrame)
        ]
        self.assertNotIn("call_old", result_ids)
        cancel_ids = [
            frame.tool_call_id
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallCancelFrame)
        ]
        self.assertIn("call_old", cancel_ids)

    async def test_round_limit_synthesizes_terminal_tool_rows_and_allows_one_closure_pass(self) -> None:
        service = self._make_service(bash_tool_max_rounds=1)
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Keep calling tools."}])
        _, service_frames, _ = self._attach_assistant(service, context, auto_reenter=True)
        payloads: list[dict[str, Any]] = []
        executed_codes: list[str] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            if len(payloads) == 2:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"ls"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("closure answer")
            return ChatCompletionPassResult(output_text="closure answer", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            executed_codes.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 3
            and context.get_messages()[-1] == {"role": "assistant", "content": "closure answer"}
        )

        result_frames = [
            frame
            for frame, direction in service_frames
            if direction is FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallResultFrame)
        ]
        statuses = [frame.result["status"] for frame in result_frames]
        self.assertEqual(executed_codes, ["pwd"])
        self.assertIn("round_limit_reached", statuses)
        self.assertEqual(len(payloads), 3)

    async def test_bot_speaking_defers_tool_result_reentry_until_bot_stopped_speaking(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Wait for speech."}])
        assistant, service_frames, assistant_frames = self._attach_assistant(
            service,
            context,
            auto_reenter=False,
        )
        await assistant.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        async def fake_stream(payload, **kwargs):
            return ChatCompletionPassResult(
                output_text="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                    }
                ],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: any(
                isinstance(frame, FunctionCallResultFrame) and direction is FrameDirection.DOWNSTREAM
                for frame, direction in service_frames
            )
        )
        self.assertFalse(
            any(isinstance(frame, LLMContextFrame) for frame, _ in assistant_frames)
        )

        await assistant.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: any(isinstance(frame, LLMContextFrame) for frame, _ in assistant_frames)
        )

    async def test_generation_task_completes_before_tool_followup(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Use a tool."}])
        followup_generation_done: list[bool] = []

        def on_assistant_push(frame, direction: FrameDirection):
            if isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM:
                followup_generation_done.append(
                    service._generation_task is None or service._generation_task.done()
                )

        self._attach_assistant(
            service,
            context,
            auto_reenter=False,
            on_assistant_push=on_assistant_push,
        )

        async def fake_stream(payload, **kwargs):
            return ChatCompletionPassResult(
                output_text="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                    }
                ],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(lambda: followup_generation_done == [True])

    async def test_function_call_frames_route_through_parallel_pipeline(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "route"}])
        function_result = FunctionCallResultFrame(
            function_name="run_bash",
            tool_call_id="call_route_1",
            arguments={"code": "pwd"},
            result={"ok": True, "status": "success"},
            run_llm=True,
        )
        tts = _RecordingPassthroughProcessor()
        tts_forwarded: list[tuple[Any, FrameDirection]] = []

        async def capture_tts_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            tts_forwarded.append((frame, direction))

        tts.push_frame = capture_tts_frame  # type: ignore[method-assign]
        await tts.process_frame(function_result, FrameDirection.DOWNSTREAM)

        output = _RecordingOutputTransport()
        output_forwarded: list[tuple[Any, FrameDirection]] = []

        async def capture_output_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            output_forwarded.append((frame, direction))

        output.push_frame = capture_output_frame  # type: ignore[method-assign]
        await output.process_frame(LLMContextFrame(context), FrameDirection.UPSTREAM)

        user_aggregator = _RecordingUserAggregator(context)
        audio_collector = _RecordingAudioCollector(context=context, user_aggregator=user_aggregator)
        inert_forwarded: list[tuple[Any, FrameDirection]] = []

        async def capture_inert_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            inert_forwarded.append((frame, direction))

        user_aggregator.push_frame = capture_inert_frame  # type: ignore[method-assign]
        audio_collector.push_frame = capture_inert_frame  # type: ignore[method-assign]
        before_messages = copy.deepcopy(context.get_messages())
        function_started = FunctionCallsStartedFrame(
            function_calls=[
                FunctionCallFromLLM(
                    function_name="run_bash",
                    tool_call_id="call_route_1",
                    arguments={"code": "pwd"},
                    context=context,
                )
            ]
        )
        await audio_collector.process_frame(function_started, FrameDirection.UPSTREAM)
        await user_aggregator.process_frame(function_started, FrameDirection.UPSTREAM)

        parallel = ParallelPipeline(
            [_RecordingPassthroughProcessor()],
            [_RecordingPassthroughProcessor()],
        )
        deduped_frames: list[tuple[Any, FrameDirection]] = []

        async def capture_parallel_frame(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            deduped_frames.append((frame, direction))

        parallel.push_frame = capture_parallel_frame  # type: ignore[method-assign]
        followup_frame = LLMContextFrame(context)
        await parallel._parallel_push_frame(followup_frame, FrameDirection.UPSTREAM)
        await parallel._parallel_push_frame(followup_frame, FrameDirection.UPSTREAM)

        self.assertEqual(tts_forwarded, [(function_result, FrameDirection.DOWNSTREAM)])
        self.assertEqual(
            [(type(frame), direction) for frame, direction in output_forwarded],
            [(LLMContextFrame, FrameDirection.UPSTREAM)],
        )
        self.assertEqual(before_messages, context.get_messages())
        self.assertEqual(len(inert_forwarded), 2)
        self.assertEqual(
            [(type(frame), direction) for frame, direction in deduped_frames],
            [(LLMContextFrame, FrameDirection.UPSTREAM)],
        )

    async def test_tool_round_via_bot_pipeline_shape(self) -> None:
        context = self._context_with_messages([{"role": "user", "content": "pipeline"}])
        user_aggregator = _RecordingUserAggregator(context)
        audio_collector = _RecordingAudioCollector(context=context, user_aggregator=user_aggregator)
        llm = _DummySerialToolRoundLLM(context, ["pwd", "ls"])
        tts = _RecordingPassthroughProcessor()
        stt = _RecordingPassthroughProcessor()
        output = _RecordingOutputTransport()
        assistant = _DummyToolFollowupAssistant(context)
        recorded_output_frames: list[tuple[Any, FrameDirection]] = []

        async def user_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            return None

        async def audio_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            await user_aggregator.process_frame(frame, direction)

        async def llm_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            if direction is FrameDirection.DOWNSTREAM:
                await tts.process_frame(frame, direction)
            else:
                await audio_collector.process_frame(frame, direction)

        async def tts_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            await output.process_frame(frame, direction)

        async def output_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            recorded_output_frames.append((frame, direction))
            if direction is FrameDirection.DOWNSTREAM:
                await assistant.process_frame(frame, direction)
            else:
                await llm.process_frame(frame, direction)
                await stt.process_frame(frame, direction)

        async def assistant_push(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
            if direction is FrameDirection.UPSTREAM:
                await output.process_frame(frame, direction)

        user_aggregator.push_frame = user_push  # type: ignore[method-assign]
        audio_collector.push_frame = audio_push  # type: ignore[method-assign]
        llm.push_frame = llm_push  # type: ignore[method-assign]
        tts.push_frame = tts_push  # type: ignore[method-assign]
        output.push_frame = output_push  # type: ignore[method-assign]
        assistant.push_frame = assistant_push  # type: ignore[method-assign]

        initial_frame = LLMContextFrame(context)
        await llm.process_frame(initial_frame, FrameDirection.DOWNSTREAM)
        await stt.process_frame(initial_frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(llm.initial_requests, 1)
        self.assertEqual(llm.executed_commands, ["pwd", "ls"])
        self.assertEqual(llm.followup_requests, 1)
        self.assertEqual(assistant.followup_pushes, 1)
        self.assertTrue(
            any(
                isinstance(frame, LLMContextFrame) and direction is FrameDirection.UPSTREAM
                for frame, direction in stt.seen_frames
            )
        )
        self.assertTrue(
            any(
                isinstance(frame, FunctionCallResultFrame) and direction is FrameDirection.DOWNSTREAM
                for frame, direction in recorded_output_frames
            )
        )


if __name__ == "__main__":
    unittest.main()
