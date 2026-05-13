"""Deterministic repro for the conversation_id rotation on interrupted tool turns.

See proj-2026-05-11-2008/conversation-cache-rotation-issue.md for the full
write-up. Run it with:

    PYTHONPATH=src .venv-pipecat/bin/python -m pytest \
        proj-2026-05-11-2008/repro_conversation_cache_rotation.py -s -v

What this reproduces
--------------------
`NemotronOmniAudioLLMService.committed_messages` is the client's mirror of "what
messages the conversation-cache checkpoint covers". `_process_context` snapshots
it from the full transcript that was just sent
(`nemotron_omni.py: self.committed_messages = copy.deepcopy(full_messages)`).
A tool *re-entry* request carries the provisional `assistant(tool_calls)` and
`tool(result)` rows that `NemotronAssistantAggregator` added to the live context.
If the re-entry's response completes before the bot gets interrupted, that
snapshot has already captured those provisional rows. When the interruption then
removes them (`NemotronAssistantAggregator._drop_all_provisional_sync_tool_rows`)
and a new user turn arrives, the live transcript is no longer a prefix-extension
of `committed_messages`, so `_build_payload` rotates the `conversation_id`.

This script sets the service into the post-re-entry state (committed_messages
includes the provisional rows), then feeds the post-interruption transcript
through `_build_payload` -- the exact call `_process_context` makes -- and shows
the rotation happen.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

# Make `import nemotron_voice...` work when run from the repo root without
# PYTHONPATH (the pytest invocation above sets it; this is a belt-and-braces).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402

from nemotron_voice.services.nvidia.nemotron_omni import (  # noqa: E402
    NemotronOmniAudioLLMService,
)


SYSTEM_INSTRUCTION = "You are a terse assistant."

# The structured run_bash result object the model sees as the `tool` row content.
TOOL_RESULT_JSON = (
    '{"ok": true, "status": "success", "summary": "Command completed.", '
    '"command": "pwd", "exit_code": 0, "timed_out": false, '
    '"stdout": "/repo\\n", "stderr": ""}'
)

# The provisional rows NemotronAssistantAggregator adds to the live LLMContext
# during a sync-tool pass (assistant(tool_calls) with empty spoken text, plus the
# tool row -- here already filled in with the result, as it would be by the time
# the re-entry request goes out).
ASSISTANT_TOOL_CALL_ROW = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "run_bash", "arguments": '{"code":"pwd"}'},
        }
    ],
}
TOOL_RESULT_ROW = {
    "role": "tool",
    "content": TOOL_RESULT_JSON,
    "tool_call_id": "call_1",
}


def _make_service() -> NemotronOmniAudioLLMService:
    service = NemotronOmniAudioLLMService(
        model="repro-model",
        conversation_id="repro-conv",
        enable_bash_tool=True,
    )
    service._settings.system_instruction = SYSTEM_INSTRUCTION
    return service


def test_committed_prefix_diverges_after_interrupted_tool_turn() -> None:
    service = _make_service()
    assert service._conversation_id == "repro-conv"

    # --- (A) Reconstruct the state right after a successful tool re-entry -------
    #
    # In a live session:
    #   1. user "Run pwd"  ->  model replies with a tool call (no spoken text)
    #      -> _process_context sets committed_messages = [system, user("Run pwd")]
    #   2. NemotronAssistantAggregator appends ASSISTANT_TOOL_CALL_ROW and
    #      TOOL_RESULT_ROW to the live LLMContext (provisional rows).
    #   3. tool runs; the tool row's content is filled in with the result.
    #   4. tool follow-up (re-entry) request goes out; its full transcript is
    #      [system, user("Run pwd"), ASSISTANT_TOOL_CALL_ROW, TOOL_RESULT_ROW];
    #      the response completes -> _process_context sets
    #          committed_messages = [system, user("Run pwd"),
    #                                ASSISTANT_TOOL_CALL_ROW, TOOL_RESULT_ROW]
    #
    # We build (1)'s normalized transcript via the real code path so the leading
    # rows are exactly what the normalizer produces, then append the two
    # provisional rows to get (4)'s committed_messages.
    pre_tool_context = LLMContext(messages=[{"role": "user", "content": "Run pwd"}])
    pre_tool_snapshot = service._normalized_request_snapshot(pre_tool_context)
    assert pre_tool_snapshot is not None
    built = service._build_payload(pre_tool_snapshot)
    assert built is not None
    _payload_1, normalized_full_1 = built
    # normalized_full_1 == [{"role":"system",...}, {"role":"user","content":"Run pwd"}]
    assert [m["role"] for m in normalized_full_1] == ["system", "user"]

    committed_after_reentry = normalized_full_1 + [
        copy.deepcopy(ASSISTANT_TOOL_CALL_ROW),
        copy.deepcopy(TOOL_RESULT_ROW),
    ]
    service.committed_messages = copy.deepcopy(committed_after_reentry)
    service._conversation_cache_committed = True
    # Pin the committed cache-shape fingerprint to the request shape so the only
    # thing that can trigger a rotation below is the prefix mismatch, not a
    # cache-shape change.
    service._committed_cache_shape_fingerprint = pre_tool_snapshot.cache_shape_fingerprint

    # --- (B) The interruption: NemotronAssistantAggregator drops the provisional
    # rows from the live context, and the interrupting user turn is appended. ----
    #
    # Live transcript is now just:
    #   [system, user("Run pwd"), user("What did the command print?")]
    # i.e. the assistant(tool_calls) + tool rows are gone.
    post_interruption_context = LLMContext(
        messages=[
            {"role": "user", "content": "Run pwd"},
            {"role": "user", "content": "What did the command print?"},
        ]
    )
    post_interruption_snapshot = service._normalized_request_snapshot(post_interruption_context)
    assert post_interruption_snapshot is not None
    # Same request shape as before -> fingerprint matches -> no shape-change rotation.
    assert (
        post_interruption_snapshot.cache_shape_fingerprint
        == service._committed_cache_shape_fingerprint
    )

    # --- (C) Build the next request. This is the exact call _process_context
    # makes; it is where the divergence is detected and the rotation happens. ----
    conversation_id_before = service._conversation_id
    print(f"\n[repro] conversation_id before next request: {conversation_id_before}")
    print(f"[repro] committed_messages ({len(service.committed_messages)} rows):")
    for i, m in enumerate(service.committed_messages):
        print(f"          [{i}] {NemotronOmniAudioLLMService._compact_message_repr(m)}")
    # `current_full` that _build_payload will compute internally:
    current_full_preview = service._with_system_message(post_interruption_snapshot.messages)
    print(f"[repro] current_full ({len(current_full_preview)} rows):")
    for i, m in enumerate(current_full_preview):
        print(f"          [{i}] {NemotronOmniAudioLLMService._compact_message_repr(m)}")

    built_2 = service._build_payload(post_interruption_snapshot)
    assert built_2 is not None
    payload_2, _normalized_full_2 = built_2

    conversation_id_after = service._conversation_id
    print(f"[repro] conversation_id after next request:  {conversation_id_after}")
    print(f"[repro] payload conversation_id: {payload_2.get('conversation_id')!r}")
    print(
        f"[repro] payload has conversation_require_cache? "
        f"{'conversation_require_cache' in payload_2}"
    )
    print(f"[repro] payload sent {len(payload_2['messages'])} messages (full-history rebase)")

    # --- Assertions: the rotation happened, and the request went out as a
    # full-history rebase (no cache reuse). -----------------------------------
    assert conversation_id_after != conversation_id_before, (
        "expected a conversation_id rotation because committed_messages "
        f"({[m['role'] for m in service.committed_messages]}) "  # noqa: E501 -- diagnostic only
        "is no longer a prefix of the live transcript "
        f"({[m['role'] for m in current_full_preview]})"
    )
    assert "conversation_require_cache" not in payload_2, (
        "after rotation the request must go out as a full-history rebase, "
        "not a cache-reusing suffix"
    )
    # And the committed mirror is reset (it will be re-established from this
    # rebase's response).
    assert service.committed_messages == []
    assert service._conversation_cache_committed is False


if __name__ == "__main__":  # pragma: no cover - convenience for `python repro_...py`
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
