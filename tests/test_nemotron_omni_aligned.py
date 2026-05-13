import asyncio
import contextlib
import copy
import functools
import json
import subprocess
import sys
import unittest
from unittest import mock
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VLLM_PROJECT_ROOT = _REPO_ROOT / "vllm-v0.20.0"
_NEMOTRON_MODEL_DIR = (
    _REPO_ROOT / "models" / "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)

from nemotron_voice.bot import (  # noqa: E402
    AudioOnlyLLMUserAggregator,
    UserAudioContextCollector,
    _build_llm_context,
)
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    ChatCompletionPassResult,
    ConversationCacheMissError,
    DEFAULT_VOICE_SYSTEM_INSTRUCTION,
    InterruptedToolPassSignal,
    NemotronAssistantAggregator,
    NemotronExactAssistantMessageFrame,
    NemotronOmniAudioLLMService,
)


def _vllm_render_runner_command() -> list[str]:
    project_python = _VLLM_PROJECT_ROOT / ".venv" / "bin" / "python"
    if project_python.exists():
        return [str(project_python), "-"]
    return ["uv", "run", "python", "-"]


@functools.lru_cache(maxsize=32)
def _render_nemotron_tokens_via_vllm_subprocess(
    messages_json: str,
    add_generation_prompt: bool,
    tools_json: str | None,
    chat_template_kwargs_json: str,
) -> tuple[int, ...]:
    script = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, {json.dumps(str(_VLLM_PROJECT_ROOT))})

from transformers import AutoTokenizer
from vllm.entrypoints.chat_utils import _postprocess_messages

model_dir = Path({json.dumps(str(_NEMOTRON_MODEL_DIR))})
messages = json.loads({messages_json!r})
tools_json = {tools_json!r}
tools = json.loads(tools_json) if tools_json is not None else None
chat_template_kwargs = json.loads({chat_template_kwargs_json!r})

_postprocess_messages(messages)
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir),
    trust_remote_code=True,
    fix_mistral_regex=True,
)
chat_template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
token_ids = tokenizer.apply_chat_template(
    conversation=messages,
    tools=tools,
    chat_template=chat_template,
    tokenize=True,
    return_dict=False,
    add_generation_prompt={str(add_generation_prompt)},
    **chat_template_kwargs,
)
print(json.dumps(token_ids))
""".strip()

    completed = subprocess.run(
        _vllm_render_runner_command(),
        cwd=_VLLM_PROJECT_ROOT,
        input=script,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise unittest.SkipTest(
            "Nemotron tokenizer/chat-template render helper unavailable in this env: "
            f"{completed.stderr.strip() or completed.stdout.strip() or completed.returncode}"
        )
    return tuple(json.loads(completed.stdout.strip()))


def _render_nemotron_tokens(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    return _render_nemotron_tokens_via_vllm_subprocess(
        json.dumps(messages, ensure_ascii=True, sort_keys=True),
        add_generation_prompt,
        (
            json.dumps(tools, ensure_ascii=True, sort_keys=True)
            if tools is not None
            else None
        ),
        json.dumps(
            chat_template_kwargs or {"enable_thinking": False},
            ensure_ascii=True,
            sort_keys=True,
        ),
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
        conversation_id: str | None = None,
        strip_historical_audio_from_payload: bool | None = None,
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(
            enable_bash_tool=enable_bash_tool,
            bash_tool_max_rounds=bash_tool_max_rounds,
            conversation_id=conversation_id,
        )
        service._settings.system_instruction = system_instruction
        if strip_historical_audio_from_payload is not None:
            service._strip_historical_audio_from_payload = strip_historical_audio_from_payload

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

    def _normalized_full_messages(
        self,
        service: NemotronOmniAudioLLMService,
        context: LLMContext,
    ) -> tuple[Any, list[dict[str, Any]]]:
        snapshot = service._normalized_request_snapshot(copy.deepcopy(context))
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        full_messages = service._with_system_message(snapshot.messages)
        if service._strip_historical_audio_from_payload:
            full_messages = service._strip_historical_audio_from_messages(full_messages)
        return snapshot, full_messages

    def _expected_payload(
        self,
        service: NemotronOmniAudioLLMService,
        *,
        snapshot: Any,
        messages: list[dict[str, Any]],
        conversation_id: str | None,
        require_cache: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": service._settings.model,
            "messages": copy.deepcopy(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "_cache_shape_fingerprint": snapshot.cache_shape_fingerprint,
        }
        if snapshot.tools is not None:
            payload["tools"] = copy.deepcopy(snapshot.tools)
            if snapshot.tool_choice is not None:
                payload["tool_choice"] = copy.deepcopy(snapshot.tool_choice)
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
            if require_cache:
                payload["conversation_require_cache"] = True
        if service._settings.max_tokens is not None:
            payload["max_tokens"] = service._settings.max_tokens
        if service._settings.temperature is not None:
            payload["temperature"] = service._settings.temperature
        if service._settings.top_p is not None:
            payload["top_p"] = service._settings.top_p
        if service._settings.top_k is not None:
            payload["top_k"] = service._settings.top_k
        if service._settings.frequency_penalty is not None:
            payload["frequency_penalty"] = service._settings.frequency_penalty
        if service._settings.presence_penalty is not None:
            payload["presence_penalty"] = service._settings.presence_penalty
        if service._settings.seed is not None:
            payload["seed"] = service._settings.seed
        if service._settings.chat_template_kwargs:
            payload["chat_template_kwargs"] = copy.deepcopy(
                service._settings.chat_template_kwargs
            )
        if service._settings.extra:
            payload.update(copy.deepcopy(service._settings.extra))
        return payload

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
        use_stock_assistant: bool = False,
        interrupted_tool_pass_signal: InterruptedToolPassSignal | None = None,
    ) -> tuple[LLMAssistantAggregator, list[tuple[Any, FrameDirection]], list[tuple[Any, FrameDirection]]]:
        assistant: LLMAssistantAggregator
        if use_stock_assistant:
            assistant = LLMAssistantAggregator(context)
        else:
            assistant = NemotronAssistantAggregator(
                context,
                interrupted_tool_pass_signal=interrupted_tool_pass_signal,
            )
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

    async def test_sync_tool_round_commits_single_exact_assistant_message_and_tool_rows_as_one_batch(
        self,
    ) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Inspect two things."}])
        _, service_frames, assistant_frames = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Checking the workspace.")
            return ChatCompletionPassResult(
                output_text="Checking the workspace.",
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
                ],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
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
            lambda: len([m for m in context.get_messages() if m.get("role") == "tool"]) == 2
        )

        assistant_rows = [m for m in context.get_messages() if m.get("role") == "assistant"]
        tool_rows = [m for m in context.get_messages() if m.get("role") == "tool"]
        self.assertEqual(len(assistant_rows), 1)
        self.assertEqual(assistant_rows[0]["content"], "Checking the workspace.")
        self.assertEqual(len(assistant_rows[0]["tool_calls"]), 2)
        self.assertEqual([json.loads(row["content"])["stdout"] for row in tool_rows], ["pwd", "ls"])
        self.assertNotIn(
            {"role": "assistant", "content": "Checking the workspace."},
            context.get_messages(),
        )
        self.assertTrue(all(row["content"] != "IN_PROGRESS" for row in tool_rows))
        self.assertEqual(
            [type(frame) for frame, _ in assistant_frames if isinstance(frame, LLMContextAssistantTimestampFrame)],
            [LLMContextAssistantTimestampFrame],
        )
        self.assertTrue(
            any(
                isinstance(frame, NemotronExactAssistantMessageFrame)
                for frame, direction in service_frames
                if direction is FrameDirection.DOWNSTREAM
            )
        )

    async def test_shared_context_mixed_assistant_message_matches_stream(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Use bash and narrate."}])
        _, service_frames, _ = self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Let me ")
            await service._push_llm_text("check")
            return ChatCompletionPassResult(
                output_text="Let me check",
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
                "summary": "ran pwd",
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
            lambda: any(message.get("role") == "tool" for message in context.get_messages())
        )

        assistant_row = next(
            message for message in context.get_messages() if message.get("role") == "assistant"
        )
        self.assertEqual(assistant_row["content"], "Let me check")
        self.assertEqual(
            [frame.text for frame, direction in service_frames if direction is FrameDirection.DOWNSTREAM and isinstance(frame, LLMTextFrame)],
            ["Let me ", "check"],
        )
        self.assertEqual(
            assistant_row["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                }
            ],
        )

    async def test_mixed_pass_does_not_double_commit_assistant_row(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Narrate then use bash."}])
        self._attach_assistant(service, context)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Checking now.")
            return ChatCompletionPassResult(
                output_text="Checking now.",
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
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: any(message.get("role") == "tool" for message in context.get_messages())
        )

        assistant_rows = [m for m in context.get_messages() if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_rows), 1)
        self.assertEqual(assistant_rows[0]["content"], "Checking now.")
        self.assertIn("tool_calls", assistant_rows[0])

    async def test_sync_tool_handler_context_mutation_survives_followup_projection(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Original prompt"}])
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
            await service._push_llm_text("done")
            return ChatCompletionPassResult(output_text="done", tool_calls=[], first_token=False)

        async def mutate_live_context(params):
            params.context.get_messages()[0]["content"] = "Mutated prompt"
            params.context.add_message({"role": "assistant", "content": "handler context edit"})
            if service._mark_batch_ready_for_followup(params.tool_call_id):
                await service._await_generation_task_before_tool_followup()
            await params.result_callback(
                {
                    "ok": True,
                    "status": "success",
                    "summary": "done",
                    "command": "pwd",
                    "exit_code": 0,
                    "timed_out": False,
                    "stdout": "/repo\n",
                    "stderr": "",
                }
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service.register_function("run_bash", mutate_live_context, cancel_on_interruption=True)
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 2
            and context.get_messages()[-1] == {"role": "assistant", "content": "done"}
        )

        self.assertIn({"role": "user", "content": "Mutated prompt"}, payloads[1]["messages"])
        self.assertIn(
            {"role": "assistant", "content": "handler context edit"},
            payloads[1]["messages"],
        )

    async def test_assistant_turn_contract_preserved_with_exact_assistant_appends(self) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Explain while using bash."}])
        assistant, _, assistant_frames = self._attach_assistant(service, context)
        stopped_messages = []

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(_, message):
            stopped_messages.append(message)

        async def fake_stream(payload, **kwargs):
            await service._push_llm_text("Let me check.")
            return ChatCompletionPassResult(
                output_text="Let me check.",
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
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: any(message.get("role") == "tool" for message in context.get_messages())
        )

        timestamp_frames = [
            frame for frame, _ in assistant_frames if isinstance(frame, LLMContextAssistantTimestampFrame)
        ]
        self.assertEqual(len(timestamp_frames), 1)
        self.assertEqual(len(stopped_messages), 1)
        self.assertEqual(stopped_messages[0].content, "Let me check.")
        self.assertFalse(stopped_messages[0].interrupted)

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

    async def test_function_call_cancel_during_provisional_pass_drops_staged_batch_and_clears_in_progress(
        self,
    ) -> None:
        context = self._context_with_messages([{"role": "user", "content": "Cancel a tool pass."}])
        assistant = NemotronAssistantAggregator(context)
        assistant.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)  # type: ignore[method-assign]

        async def cancel_task(task, timeout=None):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assistant.cancel_task = cancel_task  # type: ignore[method-assign]
        assistant.push_frame = lambda frame, direction=FrameDirection.DOWNSTREAM: asyncio.sleep(0)  # type: ignore[method-assign]

        function_calls = [
            FunctionCallFromLLM(
                function_name="run_bash",
                tool_call_id="call_1",
                arguments={"code": "pwd"},
                context=context,
            ),
            FunctionCallFromLLM(
                function_name="run_bash",
                tool_call_id="call_2",
                arguments={"code": "ls"},
                context=context,
            ),
        ]
        await assistant.process_frame(
            NemotronExactAssistantMessageFrame(
                message={
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
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
                    ],
                }
            ),
            FrameDirection.DOWNSTREAM,
        )
        await assistant.process_frame(
            FunctionCallsStartedFrame(function_calls=function_calls),
            FrameDirection.DOWNSTREAM,
        )
        await assistant.process_frame(
            FunctionCallInProgressFrame(
                function_name="run_bash",
                tool_call_id="call_1",
                arguments={"code": "pwd"},
                cancel_on_interruption=True,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await assistant.process_frame(
            FunctionCallCancelFrame(function_name="run_bash", tool_call_id="call_2"),
            FrameDirection.DOWNSTREAM,
        )
        await assistant.process_frame(
            FunctionCallCancelFrame(function_name="run_bash", tool_call_id="call_1"),
            FrameDirection.DOWNSTREAM,
        )

        self.assertNotIn("call_1", assistant._function_calls_in_progress)
        self.assertNotIn("call_2", assistant._function_calls_in_progress)
        # Both sync-tool calls were cancelled, so the whole staged batch — the
        # exact assistant(tool_calls) row AND its provisional tool rows — must be
        # gone (a tool_calls row with no matching tool rows is an invalid
        # transcript shape).
        self.assertEqual(
            context.get_messages(),
            [{"role": "user", "content": "Cancel a tool pass."}],
        )

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

    async def test_interrupted_sync_tool_turn_drops_provisional_batch_and_appends_new_user_row(
        self,
    ) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "First turn"}])
        signal = InterruptedToolPassSignal()
        self._attach_assistant(
            service,
            context,
            auto_reenter=True,
            interrupted_tool_pass_signal=signal,
        )
        followup_started = asyncio.Event()

        async def fake_stream(payload, **kwargs):
            if payload["messages"][-1]["content"] == "First turn":
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

            await service._push_llm_text("Partial answer")
            followup_started.set()
            await asyncio.Future()

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await followup_started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: signal.replace_interrupted_tool_pass
            and context.get_messages() == [{"role": "user", "content": "First turn"}]
        )

        context.add_message({"role": "user", "content": "Second turn"})
        self.assertEqual(
            context.get_messages(),
            [
                {"role": "user", "content": "First turn"},
                {"role": "user", "content": "Second turn"},
            ],
        )

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

    async def test_interrupted_text_pass_commits_partial_assistant_row_then_next_turn_uses_assistant_partial_plus_user_suffix(
        self,
    ) -> None:
        service = self._make_service()
        context = self._context_with_messages([{"role": "user", "content": "First turn"}])
        self._attach_assistant(service, context, auto_reenter=False)
        payloads: list[dict[str, Any]] = []
        first_response_started = asyncio.Event()

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                await service._push_llm_text("Partial")
                first_response_started.set()
                await asyncio.Future()
            await service._push_llm_text("second answer")
            return ChatCompletionPassResult(
                output_text="second answer",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await first_response_started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: context.get_messages()
            == [
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "Partial"},
            ]
        )

        context.add_message({"role": "user", "content": "Second turn"})
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 2
            and context.get_messages()[-1] == {"role": "assistant", "content": "second answer"}
        )

        self.assertEqual(
            payloads[1]["messages"][-2:],
            [
                {"role": "assistant", "content": "Partial"},
                {"role": "user", "content": "Second turn"},
            ],
        )

    async def test_cached_next_turn_payload_after_committed_turn_uses_assistant_plus_user_suffix_under_client_authoritative_protocol(
        self,
    ) -> None:
        service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-text",
        )
        context = self._context_with_messages(
            [{"role": "user", "content": "First turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                await service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )
            await service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)
        self.assertEqual(
            service.committed_messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "First turn"},
            ],
        )

        context.add_message({"role": "user", "content": "Second turn"})
        await self._run_context_frame(service, context)

        self.assertEqual(payloads[0]["conversation_id"], "conv-text")
        self.assertNotIn("conversation_require_cache", payloads[0])
        self.assertEqual(payloads[1]["conversation_id"], "conv-text")
        self.assertIs(payloads[1]["conversation_require_cache"], True)
        self.assertEqual(
            payloads[1]["messages"],
            [
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )
        self.assertEqual(
            service.committed_messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )

    async def test_tool_followup_suffix_payload_after_committed_tool_turn_uses_assistant_plus_tool_suffix_under_conversation_reuse(
        self,
    ) -> None:
        service = self._make_service(
            system_instruction="sys",
            conversation_id="conv-tool",
        )
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Run pwd."}])
        self._attach_assistant(service, context, auto_reenter=True)
        payloads: list[dict[str, Any]] = []
        tool_result = {
            "ok": True,
            "status": "success",
            "summary": "Command completed successfully.",
            "command": "pwd",
            "exit_code": 0,
            "timed_out": False,
            "stdout": "/repo\n",
            "stderr": "",
        }

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="Checking.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("Done.")
            return ChatCompletionPassResult(
                output_text="Done.",
                tool_calls=[],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            return copy.deepcopy(tool_result)

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 2
            and context.get_messages()[-1] == {"role": "assistant", "content": "Done."}
        )

        self.assertEqual(payloads[0]["conversation_id"], "conv-tool")
        self.assertNotIn("conversation_require_cache", payloads[0])
        self.assertEqual(payloads[1]["conversation_id"], "conv-tool")
        self.assertIs(payloads[1]["conversation_require_cache"], True)
        self.assertEqual(
            payloads[1]["messages"],
            [
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=True),
                    "tool_call_id": "call_1",
                },
            ],
        )
        self.assertEqual(
            service.committed_messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Run pwd."},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=True),
                    "tool_call_id": "call_1",
                },
            ],
        )

    async def test_interrupted_sync_tool_turn_preserves_conversation_id_with_append_only_user_suffix(
        self,
    ) -> None:
        service = self._make_service(
            system_instruction="sys",
            conversation_id="conv-interrupt",
        )
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "First turn"}])
        signal = InterruptedToolPassSignal()
        self._attach_assistant(
            service,
            context,
            auto_reenter=True,
            interrupted_tool_pass_signal=signal,
        )
        payloads: list[dict[str, Any]] = []
        followup_started = asyncio.Event()

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
                followup_started.set()
                await asyncio.Future()
            await service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
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
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await followup_started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: signal.replace_interrupted_tool_pass
            and context.get_messages() == [{"role": "user", "content": "First turn"}]
        )

        context.add_message({"role": "user", "content": "<user_interruption>Second turn"})
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 3
            and context.get_messages()[-1] == {"role": "assistant", "content": "Second answer"}
        )

        self.assertEqual(payloads[2]["conversation_id"], "conv-interrupt")
        self.assertIs(payloads[2]["conversation_require_cache"], True)
        self.assertEqual(
            payloads[2]["messages"],
            [{"role": "user", "content": "<user_interruption>Second turn"}],
        )

    async def test_cache_miss_rebuilds_from_full_history_not_suffix_and_rebases_same_conversation_id(
        self,
    ) -> None:
        service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-rebase",
        )
        context = self._context_with_messages(
            [{"role": "user", "content": "First turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                await service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )
            if len(payloads) == 2:
                raise ConversationCacheMissError("cache entry evicted")
            await service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)
        context.add_message({"role": "user", "content": "Second turn"})
        await self._run_context_frame(service, context)

        self.assertEqual(
            payloads[1]["messages"],
            [
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )
        self.assertEqual(payloads[1]["conversation_id"], "conv-rebase")
        self.assertIs(payloads[1]["conversation_require_cache"], True)
        self.assertEqual(
            payloads[2]["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )
        self.assertEqual(payloads[2]["conversation_id"], "conv-rebase")
        self.assertNotIn("conversation_require_cache", payloads[2])
        self.assertEqual(
            service.committed_messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )
        self.assertTrue(service._conversation_cache_committed)

    async def test_non_append_rewrite_of_client_acknowledged_prefix_rotates_conversation_id(
        self,
    ) -> None:
        service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-rotate",
        )
        context = self._context_with_messages(
            [{"role": "user", "content": "Original turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(service, context)
        payloads: list[dict[str, Any]] = []

        async def fake_stream(payload, **kwargs):
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                await service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )
            await service._push_llm_text("Rotated answer")
            return ChatCompletionPassResult(
                output_text="Rotated answer",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)
        service._turn_tool_results[("run_bash", "pwd")] = {"status": "success"}
        context.get_messages()[0]["content"] = "Rewritten turn"
        context.add_message({"role": "user", "content": "Second turn"})

        with mock.patch(
            "nemotron_voice.services.nvidia.nemotron_omni.uuid.uuid4",
            return_value=mock.Mock(hex="rotated"),
        ):
            await self._run_context_frame(service, context)

        self.assertEqual(payloads[1]["conversation_id"], "pipecat-rotated")
        self.assertNotIn("conversation_require_cache", payloads[1])
        self.assertEqual(
            payloads[1]["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Rewritten turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
        )
        self.assertEqual(service._conversation_id, "pipecat-rotated")
        self.assertEqual(service._turn_tool_results, {})

    async def test_tool_schema_or_tool_choice_change_rotates_conversation_id(self) -> None:
        async def exercise_rotation(
            *,
            label: str,
            mutate_context: Callable[[LLMContext], None],
            rotated_hex: str,
        ) -> None:
            service = self._make_service(
                system_instruction="sys",
                conversation_id=f"conv-{label}",
            )
            context = self._context_with_messages([{"role": "user", "content": "First turn"}])
            self._attach_assistant(service, context)
            payloads: list[dict[str, Any]] = []

            async def fake_stream(payload, **kwargs):
                payloads.append(copy.deepcopy(payload))
                if len(payloads) == 1:
                    await service._push_llm_text("First answer")
                    return ChatCompletionPassResult(
                        output_text="First answer",
                        tool_calls=[],
                        first_token=False,
                    )
                await service._push_llm_text("Second answer")
                return ChatCompletionPassResult(
                    output_text="Second answer",
                    tool_calls=[],
                    first_token=False,
                )

            service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
            await self._run_context_frame(service, context)
            mutate_context(context)
            context.add_message({"role": "user", "content": "Second turn"})
            with mock.patch(
                "nemotron_voice.services.nvidia.nemotron_omni.uuid.uuid4",
                return_value=mock.Mock(hex=rotated_hex),
            ):
                await self._run_context_frame(service, context)

            self.assertEqual(payloads[1]["conversation_id"], f"pipecat-{rotated_hex}")
            self.assertNotIn("conversation_require_cache", payloads[1])

        await exercise_rotation(
            label="tool-choice",
            mutate_context=lambda context: context.set_tool_choice("required"),
            rotated_hex="toolchoice",
        )

        def mutate_tool_schema(context: LLMContext) -> None:
            tools = copy.deepcopy(context.tools)
            assert isinstance(tools, ToolsSchema)
            custom_tools = copy.deepcopy(tools.custom_tools or {})
            custom_tools[AdapterType.OPENAI][0]["function"]["description"] = "Changed schema"
            tools.custom_tools = custom_tools
            context.set_tools(tools)

        await exercise_rotation(
            label="tool-schema",
            mutate_context=mutate_tool_schema,
            rotated_hex="toolschema",
        )

    async def test_chat_template_kwargs_or_prompt_shape_extra_change_rotates_conversation_id(
        self,
    ) -> None:
        async def exercise_rotation(
            *,
            label: str,
            mutate_settings: Callable[[NemotronOmniAudioLLMService], None],
            rotated_hex: str,
        ) -> None:
            service = self._make_service(
                enable_bash_tool=False,
                system_instruction="sys",
                conversation_id=f"conv-{label}",
            )
            context = self._context_with_messages(
                [{"role": "user", "content": "First turn"}],
                enable_bash_tool=False,
            )
            self._attach_assistant(service, context)
            payloads: list[dict[str, Any]] = []

            async def fake_stream(payload, **kwargs):
                payloads.append(copy.deepcopy(payload))
                if len(payloads) == 1:
                    await service._push_llm_text("First answer")
                    return ChatCompletionPassResult(
                        output_text="First answer",
                        tool_calls=[],
                        first_token=False,
                    )
                await service._push_llm_text("Second answer")
                return ChatCompletionPassResult(
                    output_text="Second answer",
                    tool_calls=[],
                    first_token=False,
                )

            service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
            await self._run_context_frame(service, context)
            mutate_settings(service)
            context.add_message({"role": "user", "content": "Second turn"})
            with mock.patch(
                "nemotron_voice.services.nvidia.nemotron_omni.uuid.uuid4",
                return_value=mock.Mock(hex=rotated_hex),
            ):
                await self._run_context_frame(service, context)

            self.assertEqual(payloads[1]["conversation_id"], f"pipecat-{rotated_hex}")
            self.assertNotIn("conversation_require_cache", payloads[1])

        await exercise_rotation(
            label="chat-template",
            mutate_settings=lambda service: setattr(
                service._settings,
                "chat_template_kwargs",
                {"enable_thinking": True},
            ),
            rotated_hex="chatshape",
        )
        await exercise_rotation(
            label="extra-shape",
            mutate_settings=lambda service: setattr(
                service._settings,
                "extra",
                {"documents": [{"title": "Context", "text": "Prompt shape changed."}]},
            ),
            rotated_hex="extrashape",
        )

    async def test_cancelled_or_failed_request_does_not_advance_committed_messages(self) -> None:
        async def prime_service(
            *,
            conversation_id: str,
        ) -> tuple[NemotronOmniAudioLLMService, LLMContext]:
            service = self._make_service(
                enable_bash_tool=False,
                system_instruction="sys",
                conversation_id=conversation_id,
            )
            context = self._context_with_messages(
                [{"role": "user", "content": "First turn"}],
                enable_bash_tool=False,
            )
            self._attach_assistant(service, context)

            async def first_turn(payload, **kwargs):
                await service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )

            service._stream_completion_pass = first_turn  # type: ignore[method-assign]
            await self._run_context_frame(service, context)
            return service, context

        service, context = await prime_service(conversation_id="conv-fail")
        committed_before_failure = copy.deepcopy(service.committed_messages)
        context.add_message({"role": "user", "content": "Second turn"})

        async def failing_stream(payload, **kwargs):
            raise RuntimeError("boom")

        service._stream_completion_pass = failing_stream  # type: ignore[method-assign]
        await self._run_context_frame(service, context)
        self.assertEqual(service.committed_messages, committed_before_failure)

        service, context = await prime_service(conversation_id="conv-cancel")
        committed_before_cancel = copy.deepcopy(service.committed_messages)
        context.add_message({"role": "user", "content": "Second turn"})
        started = asyncio.Event()

        async def blocked_stream(payload, **kwargs):
            started.set()
            await asyncio.Future()

        service._stream_completion_pass = blocked_stream  # type: ignore[method-assign]
        task = asyncio.create_task(
            service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        )
        await started.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await task
        self.assertEqual(service.committed_messages, committed_before_cancel)

    async def test_prompt_shape_golden_cases(self) -> None:
        bare_service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-golden-bare",
        )
        bare_context = self._context_with_messages(
            [{"role": "user", "content": "Bare turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(bare_service, bare_context)
        bare_payloads: list[dict[str, Any]] = []

        async def bare_stream(payload, **kwargs):
            bare_payloads.append(copy.deepcopy(payload))
            await bare_service._push_llm_text("Bare answer")
            return ChatCompletionPassResult(
                output_text="Bare answer",
                tool_calls=[],
                first_token=False,
            )

        bare_service._stream_completion_pass = bare_stream  # type: ignore[method-assign]
        await self._run_context_frame(bare_service, bare_context)
        bare_snapshot, bare_full = self._normalized_full_messages(
            bare_service,
            self._context_with_messages(
                [{"role": "user", "content": "Bare turn"}],
                enable_bash_tool=False,
            ),
        )
        self.assertEqual(
            bare_payloads[0],
            self._expected_payload(
                bare_service,
                snapshot=bare_snapshot,
                messages=bare_full,
                conversation_id="conv-golden-bare",
                require_cache=False,
            ),
        )

        cached_service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-golden-cached",
        )
        cached_context = self._context_with_messages(
            [{"role": "user", "content": "First turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(cached_service, cached_context)
        cached_payloads: list[dict[str, Any]] = []

        async def cached_stream(payload, **kwargs):
            cached_payloads.append(copy.deepcopy(payload))
            if len(cached_payloads) == 1:
                await cached_service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )
            await cached_service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
                first_token=False,
            )

        cached_service._stream_completion_pass = cached_stream  # type: ignore[method-assign]
        await self._run_context_frame(cached_service, cached_context)
        cached_context.add_message({"role": "user", "content": "Second turn"})
        expected_cached_context = self._context_with_messages(
            [
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
            enable_bash_tool=False,
        )
        cached_snapshot, expected_cached_full = self._normalized_full_messages(
            cached_service,
            expected_cached_context,
        )
        await self._run_context_frame(cached_service, cached_context)
        self.assertEqual(
            cached_payloads[1],
            self._expected_payload(
                cached_service,
                snapshot=cached_snapshot,
                messages=expected_cached_full[-2:],
                conversation_id="conv-golden-cached",
                require_cache=True,
            ),
        )

        tool_service = self._make_service(
            system_instruction="sys",
            conversation_id="conv-golden-tool",
        )
        await self._prime_service_for_tools(tool_service)
        tool_context = self._context_with_messages([{"role": "user", "content": "Run pwd."}])
        self._attach_assistant(tool_service, tool_context, auto_reenter=True)
        tool_payloads: list[dict[str, Any]] = []
        tool_result = {
            "ok": True,
            "status": "success",
            "summary": "Command completed successfully.",
            "command": "pwd",
            "exit_code": 0,
            "timed_out": False,
            "stdout": "/repo\n",
            "stderr": "",
        }

        async def tool_stream(payload, **kwargs):
            tool_payloads.append(copy.deepcopy(payload))
            if len(tool_payloads) == 1:
                return ChatCompletionPassResult(
                    output_text="Checking.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await tool_service._push_llm_text("Done.")
            return ChatCompletionPassResult(
                output_text="Done.",
                tool_calls=[],
                first_token=False,
            )

        async def run_tool(code: str, *, tool_call_id: str):
            return copy.deepcopy(tool_result)

        tool_service._stream_completion_pass = tool_stream  # type: ignore[method-assign]
        tool_service._run_bash_tool = run_tool  # type: ignore[method-assign]
        await tool_service.process_frame(LLMContextFrame(tool_context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(tool_payloads) == 2
            and tool_context.get_messages()[-1] == {"role": "assistant", "content": "Done."}
        )
        expected_tool_context = self._context_with_messages(
            [
                {"role": "user", "content": "Run pwd."},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=True),
                    "tool_call_id": "call_1",
                },
            ]
        )
        tool_snapshot, expected_tool_full = self._normalized_full_messages(
            tool_service,
            expected_tool_context,
        )
        self.assertEqual(
            tool_payloads[1],
            self._expected_payload(
                tool_service,
                snapshot=tool_snapshot,
                messages=expected_tool_full[-2:],
                conversation_id="conv-golden-tool",
                require_cache=True,
            ),
        )

        interrupted_service = self._make_service(
            system_instruction="sys",
            conversation_id="conv-golden-interrupt",
        )
        await self._prime_service_for_tools(interrupted_service)
        interrupted_context = self._context_with_messages(
            [{"role": "user", "content": "First turn"}]
        )
        interrupted_signal = InterruptedToolPassSignal()
        self._attach_assistant(
            interrupted_service,
            interrupted_context,
            auto_reenter=True,
            interrupted_tool_pass_signal=interrupted_signal,
        )
        interrupted_payloads: list[dict[str, Any]] = []
        interrupted_followup_started = asyncio.Event()

        async def interrupted_stream(payload, **kwargs):
            interrupted_payloads.append(copy.deepcopy(payload))
            if len(interrupted_payloads) == 1:
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
            if len(interrupted_payloads) == 2:
                interrupted_followup_started.set()
                await asyncio.Future()
            await interrupted_service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
                first_token=False,
            )

        async def interrupted_run_tool(code: str, *, tool_call_id: str):
            return copy.deepcopy(tool_result)

        interrupted_service._stream_completion_pass = interrupted_stream  # type: ignore[method-assign]
        interrupted_service._run_bash_tool = interrupted_run_tool  # type: ignore[method-assign]
        await interrupted_service.process_frame(
            LLMContextFrame(interrupted_context),
            FrameDirection.DOWNSTREAM,
        )
        await interrupted_followup_started.wait()
        await interrupted_service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: interrupted_signal.replace_interrupted_tool_pass
            and interrupted_context.get_messages() == [{"role": "user", "content": "First turn"}]
        )
        interrupted_context.add_message(
            {"role": "user", "content": "<user_interruption>Second turn"}
        )
        expected_interrupted_context = self._context_with_messages(
            [
                {"role": "user", "content": "First turn"},
                {"role": "user", "content": "<user_interruption>Second turn"},
            ]
        )
        interrupted_snapshot, expected_interrupted_full = self._normalized_full_messages(
            interrupted_service,
            expected_interrupted_context,
        )
        await interrupted_service.process_frame(
            LLMContextFrame(interrupted_context),
            FrameDirection.DOWNSTREAM,
        )
        await self._wait_until(
            lambda: len(interrupted_payloads) == 3
            and interrupted_context.get_messages()[-1]
            == {"role": "assistant", "content": "Second answer"}
        )
        self.assertEqual(
            interrupted_payloads[2],
            self._expected_payload(
                interrupted_service,
                snapshot=interrupted_snapshot,
                messages=expected_interrupted_full[-1:],
                conversation_id="conv-golden-interrupt",
                require_cache=True,
            ),
        )

        rebase_service = self._make_service(
            enable_bash_tool=False,
            system_instruction="sys",
            conversation_id="conv-golden-rebase",
        )
        rebase_context = self._context_with_messages(
            [{"role": "user", "content": "First turn"}],
            enable_bash_tool=False,
        )
        self._attach_assistant(rebase_service, rebase_context)
        rebase_payloads: list[dict[str, Any]] = []

        async def rebase_stream(payload, **kwargs):
            rebase_payloads.append(copy.deepcopy(payload))
            if len(rebase_payloads) == 1:
                await rebase_service._push_llm_text("First answer")
                return ChatCompletionPassResult(
                    output_text="First answer",
                    tool_calls=[],
                    first_token=False,
                )
            if len(rebase_payloads) == 2:
                raise ConversationCacheMissError("cache miss")
            await rebase_service._push_llm_text("Second answer")
            return ChatCompletionPassResult(
                output_text="Second answer",
                tool_calls=[],
                first_token=False,
            )

        rebase_service._stream_completion_pass = rebase_stream  # type: ignore[method-assign]
        await self._run_context_frame(rebase_service, rebase_context)
        rebase_context.add_message({"role": "user", "content": "Second turn"})
        expected_rebase_context = self._context_with_messages(
            [
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second turn"},
            ],
            enable_bash_tool=False,
        )
        rebase_snapshot, expected_rebase_full = self._normalized_full_messages(
            rebase_service,
            expected_rebase_context,
        )
        await self._run_context_frame(rebase_service, rebase_context)
        self.assertEqual(
            rebase_payloads[2],
            self._expected_payload(
                rebase_service,
                snapshot=rebase_snapshot,
                messages=expected_rebase_full,
                conversation_id="conv-golden-rebase",
                require_cache=False,
            ),
        )

    async def test_duplicate_bash_call_dedup_persists_across_committed_tool_followup_reentry_within_one_user_turn(
        self,
    ) -> None:
        service = self._make_service()
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "Keep checking pwd."}])
        self._attach_assistant(service, context, auto_reenter=True)
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
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("done")
            return ChatCompletionPassResult(output_text="done", tool_calls=[], first_token=False)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            executed_codes.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
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
            lambda: len(payloads) == 3
            and context.get_messages()[-1] == {"role": "assistant", "content": "done"}
        )

        tool_statuses = [
            json.loads(message["content"])["status"]
            for message in context.get_messages()
            if message.get("role") == "tool"
        ]
        self.assertEqual(executed_codes, ["pwd"])
        self.assertEqual(tool_statuses, ["success", "duplicate_suppressed"])

    async def test_interrupted_uncommitted_tool_batch_clears_dedup_and_round_accounting(
        self,
    ) -> None:
        service = self._make_service(bash_tool_max_rounds=1)
        await self._prime_service_for_tools(service)
        context = self._context_with_messages([{"role": "user", "content": "First turn"}])
        signal = InterruptedToolPassSignal()
        self._attach_assistant(
            service,
            context,
            auto_reenter=True,
            interrupted_tool_pass_signal=signal,
        )
        payloads: list[dict[str, Any]] = []
        executed_codes: list[str] = []
        blocked_closure = asyncio.Event()

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
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            if len(payloads) == 3:
                await service._push_llm_text("Partial closure")
                blocked_closure.set()
                await asyncio.Future()
            if len(payloads) == 4:
                return ChatCompletionPassResult(
                    output_text="",
                    tool_calls=[
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                        }
                    ],
                    first_token=False,
                )
            await service._push_llm_text("second turn done")
            return ChatCompletionPassResult(
                output_text="second turn done",
                tool_calls=[],
                first_token=False,
            )

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            executed_codes.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "done",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "/repo\n",
                "stderr": "",
            }

        service._stream_completion_pass = fake_stream  # type: ignore[method-assign]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await blocked_closure.wait()
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: signal.replace_interrupted_tool_pass
            and all(
                json.loads(message["content"])["status"] != "round_limit_reached"
                for message in context.get_messages()
                if message.get("role") == "tool"
            )
        )

        context.add_message({"role": "user", "content": "Second turn"})
        await service.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        await self._wait_until(
            lambda: len(payloads) == 5
            and context.get_messages()[-1] == {"role": "assistant", "content": "second turn done"}
        )

        tool_statuses = [
            json.loads(message["content"])["status"]
            for message in context.get_messages()
            if message.get("role") == "tool"
        ]
        self.assertEqual(executed_codes, ["pwd", "pwd"])
        self.assertEqual(tool_statuses, ["success", "success"])

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

    def test_committed_prompt_token_ids_equal_render_of_committed_messages_without_gen_prompt(
        self,
    ) -> None:
        tools = [copy.deepcopy(BASH_TOOL_DEFINITION)]
        committed_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "First answer"},
        ]
        next_turn_messages = [
            *committed_messages,
            {"role": "user", "content": "Second turn"},
        ]

        committed_prompt_token_ids = _render_nemotron_tokens(
            committed_messages,
            add_generation_prompt=False,
            tools=tools,
        )
        next_turn_prompt_token_ids = _render_nemotron_tokens(
            next_turn_messages,
            add_generation_prompt=True,
            tools=tools,
        )

        self.assertEqual(
            next_turn_prompt_token_ids[: len(committed_prompt_token_ids)],
            committed_prompt_token_ids,
        )

    def test_chat_template_renders_each_message_independently_of_later_messages(
        self,
    ) -> None:
        tools = [copy.deepcopy(BASH_TOOL_DEFINITION)]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Run pwd"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {
                            "name": "run_bash",
                            "arguments": '{"code":"pwd"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "name": "run_bash",
                "content": '{"ok":true,"status":"success","stdout":"/tmp\\n"}',
            },
            {"role": "assistant", "content": "/tmp"},
            {"role": "user", "content": "What exact path did it print?"},
        ]
        full_prompt_token_ids = _render_nemotron_tokens(
            messages,
            add_generation_prompt=True,
            tools=tools,
        )

        for prefix_len in range(1, len(messages) + 1):
            with self.subTest(prefix_len=prefix_len):
                prefix_prompt_token_ids = _render_nemotron_tokens(
                    messages[:prefix_len],
                    add_generation_prompt=False,
                    tools=tools,
                )
                self.assertEqual(
                    full_prompt_token_ids[: len(prefix_prompt_token_ids)],
                    prefix_prompt_token_ids,
                )

    def test_assistant_and_tool_call_history_round_trips_to_generation_tokens(
        self,
    ) -> None:
        tools = [copy.deepcopy(BASH_TOOL_DEFINITION)]
        openai_style_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Run pwd"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {
                            "name": "run_bash",
                            "arguments": '{"code":"pwd"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "name": "run_bash",
                "content": '{"ok":true,"status":"success","stdout":"/tmp\\n"}',
            },
        ]
        normalized_messages = copy.deepcopy(openai_style_messages)
        normalized_messages[2]["tool_calls"][0]["function"]["arguments"] = {
            "code": "pwd"
        }

        openai_render_tokens = _render_nemotron_tokens(
            openai_style_messages,
            add_generation_prompt=False,
            tools=tools,
        )
        normalized_render_tokens = _render_nemotron_tokens(
            normalized_messages,
            add_generation_prompt=False,
            tools=tools,
        )

        # Perf-only proxy: if the production render path changes how assistant
        # tool-call history round-trips, the first cached tool follow-up loses a
        # native prefix-cache hit even though correctness still holds.
        self.assertEqual(openai_render_tokens, normalized_render_tokens)


if __name__ == "__main__":
    unittest.main()
