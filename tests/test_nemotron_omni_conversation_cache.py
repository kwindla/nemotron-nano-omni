import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_voice.bot import AudioOnlyLLMUserAggregator  # noqa: E402
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    NemotronOmniAudioLLMService,
)


class _FakeProcess:
    def __init__(self, *, first_result, second_result=(b"", b"")):
        self._first_result = first_result
        self._second_result = second_result
        self._calls = 0
        self.returncode = None
        self.killed = False
        self.reaped = False

    async def communicate(self):
        self._calls += 1
        if self._calls == 1:
            result = self._first_result
            if asyncio.iscoroutine(result):
                return await result
            return await result()
        self.reaped = True
        return self._second_result

    def kill(self):
        self.killed = True
        self.returncode = -9


class NemotronOmniConversationCacheTests(unittest.IsolatedAsyncioTestCase):
    """Helper coverage for conversation-cache-related service behavior.

    Integration scenarios for suffix projection, rotation, and rebase live in
    ``tests/test_nemotron_omni_aligned.py``.
    """

    def _make_service(
        self,
        *,
        strip_historical_audio_from_payload: bool = False,
        bash_tool_timeout_secs: float = 0.05,
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(
            enable_bash_tool=True,
            bash_tool_timeout_secs=bash_tool_timeout_secs,
        )
        service._strip_historical_audio_from_payload = strip_historical_audio_from_payload

        async def noop(*args, **kwargs):
            return None

        service.push_frame = noop  # type: ignore[method-assign]
        service.start_processing_metrics = noop  # type: ignore[method-assign]
        service.stop_processing_metrics = noop  # type: ignore[method-assign]
        service.start_ttfb_metrics = noop  # type: ignore[method-assign]
        service.stop_ttfb_metrics = noop  # type: ignore[method-assign]
        service.start_llm_usage_metrics = noop  # type: ignore[method-assign]
        service._push_llm_text = noop  # type: ignore[method-assign]
        service._session = object()
        return service

    def test_historical_audio_stripping_defaults_to_disabled(self) -> None:
        # OFF by default: stripping is relative to the latest user row, so it
        # would make `committed_messages` non-monotonic and rotate the cache
        # every turn. Cross-turn prefix reuse requires the un-stripped transcript.
        service = NemotronOmniAudioLLMService(enable_bash_tool=True)
        self.assertFalse(service._strip_historical_audio_from_payload)

    def test_is_conversation_cache_miss_only_for_409_with_that_error_type(self) -> None:
        cls = NemotronOmniAudioLLMService
        self.assertTrue(
            cls._is_conversation_cache_miss(
                409, '{"error": {"type": "ConversationCacheMissError", "message": "evicted"}}'
            )
        )
        # wrong status
        self.assertFalse(
            cls._is_conversation_cache_miss(
                400, '{"error": {"type": "ConversationCacheMissError"}}'
            )
        )
        # 409 but a different error type
        self.assertFalse(
            cls._is_conversation_cache_miss(409, '{"error": {"type": "SomethingElse"}}')
        )
        # 409 with non-JSON / non-dict error body
        self.assertFalse(cls._is_conversation_cache_miss(409, "not json"))
        self.assertFalse(cls._is_conversation_cache_miss(409, '{"error": "string"}'))

    def test_messages_for_adapter_boundary_uses_openai_llm_specific_id(self) -> None:
        service = self._make_service()
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                LLMSpecificMessage(
                    llm="openai",
                    message={"role": "assistant", "content": "Keep this provider row."},
                ),
                LLMSpecificMessage(
                    llm="NemotronOmniAudioLLMService",
                    message={"role": "assistant", "content": "Drop this legacy provider row."},
                ),
            ]
        )

        filtered_messages = service._messages_for_adapter_boundary(context)

        self.assertEqual(
            filtered_messages,
            [
                {"role": "user", "content": "Hello"},
                LLMSpecificMessage(
                    llm="openai",
                    message={"role": "assistant", "content": "Keep this provider row."},
                ),
            ],
        )

    def test_provider_messages_downgrade_developer_to_user_before_conversion(self) -> None:
        service = self._make_service()
        provider_messages = service._provider_messages_from_universal_messages(
            [
                LLMSpecificMessage(
                    llm="openai",
                    message={"role": "developer", "content": "Developer note."},
                ),
                {"role": "user", "content": "Question"},
            ]
        )

        self.assertEqual(
            provider_messages,
            [
                {"role": "user", "content": "Developer note."},
                {"role": "user", "content": "Question"},
            ],
        )

    def test_strip_historical_audio_from_messages_preserves_latest_user_audio(self) -> None:
        service = self._make_service(strip_historical_audio_from_payload=True)
        messages = service._with_system_message(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "User audio follows."},
                        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA="}},
                    ],
                },
                {"role": "assistant", "content": "A unicorn."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "New user audio follows."},
                        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,BBB="}},
                    ],
                },
            ]
        )

        stripped = service._strip_historical_audio_from_messages(messages)

        self.assertEqual(
            stripped[1]["content"],
            [{"type": "text", "text": "User audio follows."}],
        )
        self.assertEqual(
            stripped[-1]["content"],
            [
                {"type": "text", "text": "New user audio follows."},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,BBB="}},
            ],
        )

    def test_build_bash_tool_result_returns_fixed_field_contract(self) -> None:
        service = self._make_service()
        result = service._build_bash_tool_result(
            command="pwd",
            command_started=True,
            exit_code=0,
            stdout_text="out",
            stderr_text="",
            timed_out=False,
        )

        self.assertEqual(
            set(result.keys()),
            {"ok", "status", "summary", "command", "exit_code", "timed_out", "stdout", "stderr"},
        )
        self.assertEqual(result["status"], "success")

    async def test_execute_bash_tool_request_rejects_missing_code(self) -> None:
        service = self._make_service()

        result = await service._execute_bash_tool_request(
            arguments={},
            tool_call_id="call_1",
            function_name="run_bash",
        )

        self.assertEqual(result["status"], "invalid_arguments")
        self.assertIn("Missing required string argument", result["summary"])

    async def test_execute_bash_tool_request_rejects_long_echo_prose(self) -> None:
        service = self._make_service()
        events: list[dict[str, object]] = []

        async def should_not_run_bash(*args, **kwargs):
            raise AssertionError("bash subprocess should not run for long prose echo")

        async def capture_event(payload):
            events.append(payload)

        service._run_bash_tool = should_not_run_bash  # type: ignore[method-assign]
        service._bash_tool_event_sender = capture_event  # type: ignore[assignment]

        result = await service._execute_bash_tool_request(
            arguments={
                "code": (
                    'echo "A dragon is a legendary creature often depicted as '
                    'a large, serpentine or reptilian beast with wings."'
                )
            },
            tool_call_id="call_1",
            function_name="run_bash",
        )

        self.assertEqual(result["status"], "policy_rejected")
        self.assertIn("Do not use bash to echo", result["summary"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "policy_rejected")
        self.assertIs(events[0]["guardrail_triggered"], True)

    async def test_execute_bash_tool_request_allows_short_echo_command(self) -> None:
        service = self._make_service()
        seen_calls: list[tuple[str, str]] = []
        tool_result = {
            "ok": True,
            "status": "success",
            "summary": "Command completed successfully.",
            "command": "echo spark audio one",
            "exit_code": 0,
            "timed_out": False,
            "stdout": "spark audio one\n",
            "stderr": "",
        }

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            seen_calls.append((code, tool_call_id))
            return tool_result

        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        result = await service._execute_bash_tool_request(
            arguments={"code": "echo spark audio one"},
            tool_call_id="call_1",
            function_name="run_bash",
        )

        self.assertEqual(result, tool_result)
        self.assertEqual(seen_calls, [("echo spark audio one", "call_1")])

    async def test_execute_bash_tool_request_suppresses_exact_duplicates_within_one_turn(self) -> None:
        service = self._make_service()
        events: list[dict[str, object]] = []
        seen_calls: list[str] = []

        async def capture_event(payload):
            events.append(payload)

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            seen_calls.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "Command completed successfully.",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": "tool result",
                "stderr": "",
            }

        service._bash_tool_event_sender = capture_event  # type: ignore[assignment]
        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        service._reset_turn_tool_state("turn-1")

        first = await service._execute_bash_tool_request(
            arguments={"code": "pwd"},
            tool_call_id="call_1",
            function_name="run_bash",
        )
        duplicate = await service._execute_bash_tool_request(
            arguments={"code": "pwd"},
            tool_call_id="call_2",
            function_name="run_bash",
        )

        self.assertEqual(seen_calls, ["pwd"])
        self.assertEqual(first["status"], "success")
        self.assertEqual(duplicate["status"], "duplicate_suppressed")
        self.assertEqual(duplicate["stdout"], "tool result")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "duplicate_suppressed")

    async def test_execute_bash_tool_request_allows_distinct_commands_within_one_turn(self) -> None:
        service = self._make_service()
        seen_calls: list[str] = []

        async def fake_run_bash_tool(code: str, *, tool_call_id: str):
            seen_calls.append(code)
            return {
                "ok": True,
                "status": "success",
                "summary": "Command completed successfully.",
                "command": code,
                "exit_code": 0,
                "timed_out": False,
                "stdout": code,
                "stderr": "",
            }

        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]
        service._reset_turn_tool_state("turn-1")

        first = await service._execute_bash_tool_request(
            arguments={"code": "pwd"},
            tool_call_id="call_1",
            function_name="run_bash",
        )
        second = await service._execute_bash_tool_request(
            arguments={"code": "ls"},
            tool_call_id="call_2",
            function_name="run_bash",
        )

        self.assertEqual(seen_calls, ["pwd", "ls"])
        self.assertEqual(first["command"], "pwd")
        self.assertEqual(second["command"], "ls")

    async def test_communicate_bash_process_times_out_and_kills_child(self) -> None:
        service = self._make_service(bash_tool_timeout_secs=0.01)

        async def first_call():
            await asyncio.sleep(1)
            return b"", b""

        process = _FakeProcess(first_result=first_call, second_result=(b"out", b"err"))
        stdout, stderr, timed_out = await service._communicate_bash_process(process)

        self.assertTrue(timed_out)
        self.assertTrue(process.killed)
        self.assertEqual(stdout, b"out")
        self.assertEqual(stderr, b"err")

    async def test_communicate_bash_process_cancellation_kills_and_reaps_child(self) -> None:
        service = self._make_service()
        gate = asyncio.Event()

        async def first_call():
            await gate.wait()
            return b"", b""

        process = _FakeProcess(first_result=first_call, second_result=(b"out", b"err"))
        task = asyncio.create_task(service._communicate_bash_process(process))
        await asyncio.sleep(0)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(process.killed)
        self.assertTrue(process.reaped)

    async def test_audio_only_user_aggregator_ignores_transcriptions_for_context(self) -> None:
        context = LLMContext()
        aggregator = AudioOnlyLLMUserAggregator(context)

        await aggregator._handle_transcription(
            TranscriptionFrame(
                text="hello there",
                user_id="",
                timestamp="2026-05-02T00:00:00Z",
            )
        )

        pushed = await aggregator.push_aggregation()

        self.assertEqual(pushed, "")
        self.assertEqual(context.get_messages(), [])


if __name__ == "__main__":
    unittest.main()
