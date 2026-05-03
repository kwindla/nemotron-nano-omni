import copy
import json
import sys
import unittest
from pathlib import Path

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.aggregators.llm_context import LLMContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_voice.bot import AudioOnlyLLMUserAggregator  # noqa: E402
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    ChatCompletionPassResult,
    ConversationCacheMissError,
    NemotronOmniAudioLLMService,
)


class NemotronOmniConversationCacheTests(unittest.IsolatedAsyncioTestCase):
    def _make_service(
        self,
        *,
        conversation_id: str | None = "conversation-test",
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(
            conversation_id=conversation_id,
            enable_bash_tool=True,
        )

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

    def test_payload_after_tool_calls_uses_tool_suffix_with_conversation_cache(self) -> None:
        service = self._make_service()
        payload = {
            "model": "nemotron_3_nano_omni",
            "messages": [{"role": "user", "content": "Count the files."}],
            "_conversation_full_messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Count the files."},
            ],
            "tools": [copy.deepcopy(BASH_TOOL_DEFINITION)],
            "tool_choice": "auto",
            "conversation_require_cache": True,
        }
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
            }
        ]
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "run_bash",
                "content": "/tmp\n",
            }
        ]

        next_payload = service._payload_after_tool_calls(
            payload,
            assistant_text="\n",
            tool_calls=tool_calls,
            tool_messages=tool_messages,
        )

        self.assertEqual(next_payload["tools"], payload["tools"])
        self.assertEqual(next_payload["tool_choice"], "auto")
        self.assertTrue(next_payload["conversation_require_cache"])
        self.assertEqual(next_payload["messages"], tool_messages)
        self.assertEqual(
            next_payload["_conversation_full_messages"][-2]["tool_calls"],
            tool_calls,
        )
        self.assertEqual(
            next_payload["_conversation_full_messages"][-2]["content"],
            "\n",
        )
        self.assertEqual(
            next_payload["_conversation_full_messages"][-1],
            tool_messages[0],
        )

    def test_payload_after_tool_calls_keeps_full_history_without_conversation_id(self) -> None:
        service = self._make_service(conversation_id=None)
        payload = {
            "model": "nemotron_3_nano_omni",
            "messages": [{"role": "user", "content": "Count the files."}],
            "_conversation_full_messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Count the files."},
            ],
            "tools": [copy.deepcopy(BASH_TOOL_DEFINITION)],
            "tool_choice": "auto",
        }
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
            }
        ]
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "run_bash",
                "content": "/tmp\n",
            }
        ]

        next_payload = service._payload_after_tool_calls(
            payload,
            assistant_text="\n",
            tool_calls=tool_calls,
            tool_messages=tool_messages,
        )

        self.assertEqual(next_payload["messages"][-2]["role"], "assistant")
        self.assertEqual(next_payload["messages"][-2]["tool_calls"], tool_calls)
        self.assertEqual(next_payload["messages"][-2]["content"], "\n")
        self.assertEqual(next_payload["messages"][-1], tool_messages[0])

    def test_commit_canonical_messages_preserves_system_message(self) -> None:
        service = self._make_service(conversation_id=None)
        payload = {
            "_conversation_full_messages": [
                {"role": "user", "content": "Hello"},
            ]
        }

        service._commit_canonical_messages(
            payload,
            assistant_text="world",
        )

        self.assertEqual(service._canonical_messages[0]["role"], "system")
        self.assertEqual(
            service._canonical_messages[0]["content"],
            service._settings.system_instruction,
        )
        self.assertEqual(service._canonical_messages[1]["content"], "Hello")
        self.assertEqual(service._canonical_messages[2]["content"], "world")

    async def test_execute_tool_calls_includes_tool_name(self) -> None:
        service = self._make_service()

        async def fake_execute_tool_call(tool_call):
            return "ok"

        service._execute_tool_call = fake_execute_tool_call  # type: ignore[method-assign]

        tool_messages = await service._execute_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
                }
            ]
        )

        self.assertEqual(
            tool_messages,
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "run_bash",
                    "content": "ok",
                }
            ],
        )

    async def test_execute_tool_calls_suppresses_exact_duplicates_within_one_turn(self) -> None:
        service = self._make_service()
        executed_codes: list[str] = []
        events: list[dict[str, object]] = []

        async def fake_execute_tool_call(tool_call):
            arguments = tool_call["function"]["arguments"]
            executed_codes.append(arguments)
            return "tool result"

        service._execute_tool_call = fake_execute_tool_call  # type: ignore[method-assign]
        service._bash_tool_event_sender = events.append  # type: ignore[assignment]

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
            },
        ]

        tool_messages = await service._execute_tool_calls(
            tool_calls,
            seen_tool_results_by_signature={},
        )

        self.assertEqual(executed_codes, ['{"code":"pwd"}'])
        self.assertEqual(tool_messages[0]["content"], "tool result")
        self.assertIn("duplicate tool call suppressed", tool_messages[1]["content"])
        self.assertIn("tool result", tool_messages[1]["content"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "duplicate_suppressed")
        self.assertIs(events[0]["guardrail_triggered"], True)
        self.assertEqual(events[0]["guardrail_kind"], "duplicate_tool_call")
        self.assertEqual(events[0]["guardrail_reason"], "exact_duplicate_command")

    async def test_execute_tool_calls_allows_distinct_commands_within_one_turn(self) -> None:
        service = self._make_service()
        executed_codes: list[str] = []

        async def fake_execute_tool_call(tool_call):
            arguments = tool_call["function"]["arguments"]
            executed_codes.append(arguments)
            return arguments

        service._execute_tool_call = fake_execute_tool_call  # type: ignore[method-assign]

        tool_calls = [
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
        ]

        tool_messages = await service._execute_tool_calls(
            tool_calls,
            seen_tool_results_by_signature={},
        )

        self.assertEqual(executed_codes, ['{"code":"pwd"}', '{"code":"ls"}'])
        self.assertEqual(
            [message["content"] for message in tool_messages],
            ['{"code":"pwd"}', '{"code":"ls"}'],
        )

    async def test_execute_tool_call_rejects_long_echo_prose(self) -> None:
        service = self._make_service()
        events: list[dict[str, object]] = []

        async def should_not_run_bash(*args, **kwargs):
            raise AssertionError("bash subprocess should not run for long prose echo")

        service._run_bash_tool = should_not_run_bash  # type: ignore[method-assign]
        service._bash_tool_event_sender = events.append  # type: ignore[assignment]

        result = await service._execute_tool_call(
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "arguments": json.dumps(
                        {
                            "code": (
                                'echo "A dragon is a legendary creature often depicted as '
                                'a large, serpentine or reptilian beast with wings."'
                            )
                        }
                    ),
                },
            }
        )

        self.assertIn("Do not use bash to echo", result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "policy_rejected")
        self.assertIs(events[0]["guardrail_triggered"], True)
        self.assertEqual(
            events[0]["guardrail_kind"],
            "non_instrumental_bash_tool_use",
        )
        self.assertEqual(events[0]["guardrail_reason"], "echo_or_printf_prose")

    async def test_execute_tool_call_allows_short_echo_command(self) -> None:
        service = self._make_service()
        seen_calls: list[tuple[str, str]] = []

        async def fake_run_bash_tool(code: str, *, tool_call_id: str) -> str:
            seen_calls.append((code, tool_call_id))
            return "ok"

        service._run_bash_tool = fake_run_bash_tool  # type: ignore[method-assign]

        result = await service._execute_tool_call(
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "arguments": json.dumps({"code": "echo spark audio one"}),
                },
            }
        )

        self.assertEqual(result, "ok")
        self.assertEqual(seen_calls, [("echo spark audio one", "call_1")])

    def test_suffix_only_uses_latest_text_turn_after_audio_turn(self) -> None:
        service = self._make_service()
        service._conversation_cache_committed = True

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "User audio follows."},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA"}},
                ],
            },
            {"role": "assistant", "content": "I heard the audio."},
            {"role": "user", "content": "Now answer this typed follow-up."},
        ]

        payload_messages, used_suffix_only = service._conversation_payload_messages(messages)

        self.assertTrue(used_suffix_only)
        self.assertEqual(
            payload_messages,
            [{"role": "user", "content": "Now answer this typed follow-up."}],
        )

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

    async def test_successful_turn_marks_suffix_only_ready(self) -> None:
        service = self._make_service()
        seen_payloads: list[dict] = []

        async def fake_stream_completion_pass(payload, **kwargs):
            seen_payloads.append(copy.deepcopy(payload))
            return ChatCompletionPassResult(
                output_text="done",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream_completion_pass  # type: ignore[method-assign]

        payload = {
            "model": "nemotron_3_nano_omni",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        await service._run_completion_payload(
            payload,
            request_description="test request",
            start_ttfb=False,
        )

        self.assertTrue(service._conversation_cache_committed)
        self.assertEqual(seen_payloads, [payload])

    async def test_cache_miss_recovery_disables_suffix_only_for_next_turn(self) -> None:
        service = self._make_service()
        service._conversation_cache_committed = True
        seen_payloads: list[dict] = []
        attempts = 0

        async def fake_stream_completion_pass(payload, **kwargs):
            nonlocal attempts
            seen_payloads.append(copy.deepcopy(payload))
            attempts += 1
            if attempts == 1:
                raise ConversationCacheMissError("cache miss")
            return ChatCompletionPassResult(
                output_text="recovered",
                tool_calls=[],
                first_token=False,
            )

        service._stream_completion_pass = fake_stream_completion_pass  # type: ignore[method-assign]

        full_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Original full turn"},
        ]
        payload = {
            "model": "nemotron_3_nano_omni",
            "messages": [{"role": "user", "content": "Suffix-only turn"}],
            "_conversation_full_messages": copy.deepcopy(full_messages),
            "conversation_require_cache": True,
        }

        await service._run_completion_payload(
            payload,
            request_description="suffix-only request",
            start_ttfb=False,
        )

        self.assertFalse(service._conversation_cache_committed)
        self.assertEqual(len(seen_payloads), 2)
        self.assertTrue(seen_payloads[0]["conversation_require_cache"])
        self.assertEqual(seen_payloads[0]["messages"], payload["messages"])
        self.assertNotIn("conversation_require_cache", seen_payloads[1])
        self.assertEqual(seen_payloads[1]["messages"], full_messages)


if __name__ == "__main__":
    unittest.main()
