import asyncio
import json
import sys
import unittest
from pathlib import Path

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.services.llm_service import FunctionCallParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_voice.bot import _build_llm_context  # noqa: E402
from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    BASH_TOOL_DEFINITION,
    BASH_TOOL_NAME,
    NemotronOmniAudioLLMService,
)


class NemotronOmniAlignedTests(unittest.IsolatedAsyncioTestCase):
    def _make_service(
        self,
        *,
        enable_bash_tool: bool = True,
        bash_tool_timeout_secs: float = 0.01,
    ) -> NemotronOmniAudioLLMService:
        service = NemotronOmniAudioLLMService(
            enable_bash_tool=enable_bash_tool,
            bash_tool_timeout_secs=bash_tool_timeout_secs,
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
        return service

    def _sample_result(
        self,
        *,
        command: str = "pwd",
        stdout: str = "/tmp\n",
        stderr: str = "",
        exit_code: int | None = 0,
        timed_out: bool = False,
    ) -> dict[str, object]:
        service = self._make_service()
        return service._build_bash_tool_result(
            command=command,
            command_started=True,
            exit_code=exit_code,
            stdout_text=stdout,
            stderr_text=stderr,
            timed_out=timed_out,
        )

    def test_sync_tool_result_json_contract(self) -> None:
        result = self._sample_result()
        self.assertEqual(
            list(result.keys()),
            ["ok", "status", "summary", "command", "exit_code", "timed_out", "stdout", "stderr"],
        )
        self.assertEqual(
            set(result),
            {"ok", "status", "summary", "command", "exit_code", "timed_out", "stdout", "stderr"},
        )

    def test_dedup_result_uses_reserved_status(self) -> None:
        service = self._make_service()
        original = self._sample_result(stderr="normal diagnostic output\n")

        duplicate = service._duplicate_tool_result(original)

        self.assertEqual(duplicate["status"], "duplicate_suppressed")
        self.assertEqual(duplicate["ok"], original["ok"])
        self.assertEqual(duplicate["command"], original["command"])
        self.assertEqual(duplicate["exit_code"], original["exit_code"])
        self.assertEqual(duplicate["timed_out"], original["timed_out"])
        self.assertEqual(duplicate["stdout"], original["stdout"])
        self.assertEqual(duplicate["stderr"], original["stderr"])
        self.assertIn("Reused the prior result", str(duplicate["summary"]))

    async def test_duplicate_bash_call_within_one_turn_uses_dedup_handler_path(self) -> None:
        service = self._make_service()
        executed_tool_call_ids: list[str] = []
        events: list[dict[str, object]] = []
        first_result = self._sample_result()

        async def fake_execute_tool_call(tool_call):
            executed_tool_call_ids.append(tool_call["id"])
            return json.dumps(first_result)

        async def capture_event(payload):
            events.append(payload)

        service._execute_tool_call = fake_execute_tool_call  # type: ignore[method-assign]
        service._bash_tool_event_sender = capture_event  # type: ignore[assignment]

        tool_messages = await service._execute_tool_calls(
            [
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
            ],
            seen_tool_results_by_signature={},
        )

        self.assertEqual(executed_tool_call_ids, ["call_1"])
        self.assertEqual(json.loads(tool_messages[0]["content"]), first_result)
        duplicate_payload = json.loads(tool_messages[1]["content"])
        self.assertEqual(duplicate_payload["status"], "duplicate_suppressed")
        self.assertEqual(duplicate_payload["stdout"], first_result["stdout"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "duplicate_suppressed")

    async def test_handle_run_bash_function_call_validates_missing_code(self) -> None:
        service = self._make_service()
        seen_results: list[dict[str, object]] = []

        async def result_callback(result, *, properties=None):
            seen_results.append(result)

        await service._handle_run_bash_function_call(
            FunctionCallParams(
                function_name=BASH_TOOL_NAME,
                tool_call_id="call_1",
                arguments={},
                llm=service,
                context=_build_llm_context(enable_bash_tool=True),
                result_callback=result_callback,
            )
        )

        self.assertEqual(len(seen_results), 1)
        self.assertEqual(seen_results[0]["status"], "invalid_arguments")
        self.assertEqual(seen_results[0]["command"], "")
        self.assertEqual(seen_results[0]["stdout"], "")
        self.assertEqual(seen_results[0]["stderr"], "")

    async def test_handle_run_bash_function_call_rejects_prose_echo(self) -> None:
        service = self._make_service()
        events: list[dict[str, object]] = []
        seen_results: list[dict[str, object]] = []

        async def should_not_run_bash(*args, **kwargs):
            raise AssertionError("bash subprocess should not run for prose echo")

        async def result_callback(result, *, properties=None):
            seen_results.append(result)

        async def capture_event(payload):
            events.append(payload)

        service._run_bash_tool = should_not_run_bash  # type: ignore[method-assign]
        service._bash_tool_event_sender = capture_event  # type: ignore[assignment]

        await service._handle_run_bash_function_call(
            FunctionCallParams(
                function_name=BASH_TOOL_NAME,
                tool_call_id="call_1",
                arguments={
                    "code": (
                        'echo "A dragon is a legendary creature often depicted as '
                        'a large, serpentine or reptilian beast with wings."'
                    )
                },
                llm=service,
                context=_build_llm_context(enable_bash_tool=True),
                result_callback=result_callback,
            )
        )

        self.assertEqual(len(seen_results), 1)
        self.assertEqual(seen_results[0]["status"], "policy_rejected")
        self.assertIn("Do not use bash to echo", str(seen_results[0]["summary"]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "policy_rejected")

    async def test_subprocess_timeout_cleanup_kills_and_reaps_child(self) -> None:
        service = self._make_service(bash_tool_timeout_secs=0.01)

        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.kill_called = False
                self.communicate_calls = 0

            async def communicate(self):
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    await asyncio.sleep(0.05)
                    return b"", b""
                self.returncode = -9
                return b"partial stdout", b""

            def kill(self):
                self.kill_called = True
                self.returncode = -9

        process = FakeProcess()
        stdout, stderr, timed_out = await service._communicate_bash_process(process)

        self.assertTrue(timed_out)
        self.assertTrue(process.kill_called)
        self.assertEqual(process.communicate_calls, 2)
        self.assertEqual(stdout, b"partial stdout")
        self.assertEqual(stderr, b"")

    async def test_subprocess_cancellation_cleanup_kills_and_reaps_child(self) -> None:
        service = self._make_service()

        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.kill_called = False
                self.communicate_calls = 0

            async def communicate(self):
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    raise asyncio.CancelledError()
                self.returncode = -9
                return b"", b""

            def kill(self):
                self.kill_called = True
                self.returncode = -9

        process = FakeProcess()
        with self.assertRaises(asyncio.CancelledError):
            await service._communicate_bash_process(process)

        self.assertTrue(process.kill_called)
        self.assertEqual(process.communicate_calls, 2)

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

    def test_enable_bash_tool_registers_handler(self) -> None:
        service = self._make_service(enable_bash_tool=True)

        self.assertIn(BASH_TOOL_NAME, service._functions)
        self.assertTrue(service._functions[BASH_TOOL_NAME].cancel_on_interruption)

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
