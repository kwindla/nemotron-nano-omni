# Issue Investigation Notes

## Scope

This document records the deep investigation for four defects in `PLAN.md` found during adversarial review against the Pipecat reference implementation in `./pipecat-core-code`.

## 1. Staged Sync-Tool Design vs. Live `FunctionCallParams.context`

### Finding

The current plan conflicts with a real Pipecat contract: sync tool handlers receive a live, shared, mutable `LLMContext` in `FunctionCallParams.context`, and the stock sync-tool path mutates that same shared context before the handler is invoked.

### Evidence

- `LLMService._run_function_call()` broadcasts `FunctionCallInProgressFrame` before invoking the tool handler.
  - Source: `pipecat-core-code/src/pipecat/services/llm_service.py:882-979`
- `LLMAssistantAggregator._handle_function_call_in_progress()` immediately mutates shared context when it receives that frame.
  - It appends an `assistant(tool_calls)` row.
  - For sync tools, it also appends a `tool` row with `content="IN_PROGRESS"`.
  - Source: `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1258-1292`
- `BaseOpenAILLMService._process_context()` carries the original context object through `FunctionCallFromLLM(context=context)` into `run_function_calls()`.
  - Source: `pipecat-core-code/src/pipecat/services/openai/base_llm.py:521-540`
- Reference examples use `params.context` as a live object, not a passive snapshot.
  - `save_conversation()` reads `params.context.get_messages()`.
  - `load_conversation()` calls `params.context.set_messages(...)`.
  - Source: `pipecat-core-code/examples/persistent-context/persistent-context-openai.py:66-95`
  - Same pattern appears in the OpenAI Responses example: `pipecat-core-code/examples/persistent-context/persistent-context-openai-responses.py:66-95`

### Consequence

The current staged-batch wording in `PLAN.md` is too strong:

- using a deep-copied frame-time snapshot as the sole request-building source for tool followups can ignore handler-driven context edits
- delaying all assistant/tool commits until the serial batch finishes breaks the observable stock behavior that handlers run against a context already updated with the tool call they are servicing
- if handler code mutates `params.context`, a later follow-up rebuilt from the original snapshot plus locally staged rows can overwrite or silently discard those handler mutations

### Required Plan Constraints

- The frame-time deep copy may remain the source for one in-flight HTTP request only.
- It must not remain the sole request source across sync-tool re-entry.
- `FunctionCallParams.context` must remain the live shared `LLMContext`.
- The plan must explicitly preserve handler-visible context mutations on the follow-up pass.
- If service-owned assistant/tool rows are staged provisionally, the plan must define how those provisional rows coexist with handler-driven mutations to the same live context.
- If rollback/drop semantics remain for interrupted provisional tool turns, the plan must say exactly what is dropped:
  - only service-owned provisional assistant/tool rows
  - or the entire live context mutation set for that pass
  - and how handler-owned edits are preserved or rejected

## 2. Unconditional `LLMAssistantAggregator` Replacement for Text-Only Passes

### Finding

The plan overreaches by replacing `LLMAssistantAggregator` unconditionally, including no-tool and text-only passes. The reference does not justify that scope, and stock `LLMAssistantAggregator` already owns a large amount of behavior beyond simple assistant-text appends.

### Evidence

- Stock text-only assistant commits happen exactly once through `push_aggregation()`.
  - It appends the assistant message to shared context.
  - It pushes `LLMContextFrame`.
  - It emits `LLMContextAssistantTimestampFrame`.
  - Source: `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1197-1214`
- The assistant stop path is broader than `_handle_llm_end()`.
  - `InterruptionFrame` triggers `_trigger_assistant_turn_stopped(interrupted=True)`.
  - `EndFrame` / `CancelFrame` also drain pending text.
  - `LLMAssistantPushAggregationFrame` handles TTS-driven greeting commits without an LLM response cycle.
  - Sources:
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1124-1135`
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1243-1250`
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1476-1485`
- The stock assistant aggregator also owns:
  - marker handling and stripping
  - thought aggregation and `on_assistant_thought`
  - context summarizer setup and `on_summary_applied`
  - assistant turn lifecycle events and timestamps
  - Sources:
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1505-1567`
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1070-1087`
    - `pipecat-core-code/src/pipecat/processors/aggregators/llm_response_universal.py:1597-1618`
- The OpenAI function-calling example keeps the standard aggregator pair even in a tool-enabled flow.
  - Source: `pipecat-core-code/examples/function-calling/function-calling-openai.py:125-142`

### Consequence

If the Nemotron refactor takes ownership of all assistant turns, it also takes ownership of:

- partial-text interruption semantics
- TTS greeting commit semantics
- marker stripping behavior
- thought event behavior
- summarizer plumbing
- timestamp emission
- all future upstream `LLMAssistantAggregator` behavior changes

That is a much larger maintenance surface than the sync-tool exactness problem requires.

### Required Plan Constraints

- Scope any Nemotron-specific assistant aggregator behavior narrowly to sync-tool exactness.
- Do not replace stock assistant aggregation semantics for ordinary text-only passes unless a concrete reference mismatch is proven.
- If a subclass is still used, it should delegate to stock behavior when no service-owned sync-tool exactness path is active for the current pass.
- The plan should distinguish:
  - text-only passes: stock assistant aggregation path
  - sync-tool or mixed text+tool-call passes that need special exactness handling

## 3. Queued-Stale Cancellation vs. `on_function_calls_cancelled`

### Finding

The plan correctly noticed that queued-but-never-started sync tool calls need terminal frame-level cancellation, but it currently misses Pipecat's service-level cancellation event contract.

### Evidence

- On interruption, `LLMService._handle_interruptions()` cancels registered `cancel_on_interruption=True` tools through `_cancel_function_call(...)`.
  - Source: `pipecat-core-code/src/pipecat/services/llm_service.py:512-516`
- `_cancel_function_call(...)` does two things:
  - broadcasts `FunctionCallCancelFrame`
  - calls `on_function_calls_cancelled` with `FunctionCallFromLLM` items
  - Source: `pipecat-core-code/src/pipecat/services/llm_service.py:1149-1186`
- `_cancel_function_calls_by_tool_call_id(...)` does the same for the async cancellation tool path.
  - Source: `pipecat-core-code/src/pipecat/services/llm_service.py:1106-1147`
- Reference async examples register `on_function_calls_cancelled` and consume that callback directly.
  - Source: `pipecat-core-code/examples/function-calling/function-calling-openai-async.py:106-113`
  - Source: `pipecat-core-code/examples/function-calling/function-calling-openai-async-stream.py:143-150`

### Consequence

If the Nemotron sequential-runner override only broadcasts `FunctionCallCancelFrame` for stale queued siblings:

- aggregator state may clear
- user-mute strategy state may clear
- but application-level code listening to `on_function_calls_cancelled` will silently stop receiving cancellations for those ids

That is a behavioral regression from stock Pipecat.

### Required Plan Constraints

- Queued stale-item invalidation must preserve both cancellation surfaces:
  - `FunctionCallCancelFrame`
  - `on_function_calls_cancelled`
- The plan should require constructing `FunctionCallFromLLM` items for discarded queued ids and firing the cancellation event with the same data shape stock Pipecat uses.
- The plan should say whether the event is fired:
  - per stale id
  - or once per superseded batch with all stale ids together
- The plan should prefer reusing or extracting a helper aligned with stock cancellation code rather than duplicating only part of the behavior.

## 4. Developer-Role Handling Still Diverges from `OpenAILLMService`

### Finding

The plan still defers developer-role alignment, but this is a real prompt-shape divergence from the OpenAI-compatible reference path, not just a future cleanup item.

### Evidence

- `BaseOpenAILLMService` normalizes through the OpenAI adapter and passes `convert_developer_to_user=not self.supports_developer_role`.
  - Source: `pipecat-core-code/src/pipecat/services/openai/base_llm.py:295-304`
- `OpenAILLMAdapter` converts `developer` messages to `user` when requested.
  - It does not convert them to `system`.
  - Source: `pipecat-core-code/src/pipecat/adapters/services/open_ai_adapter.py:198-218`
- The adapter tests explicitly cover `developer -> user` conversion.
  - Source: `pipecat-core-code/tests/test_get_llm_invocation_params.py:334-360`
- OpenAI-compatible services that do not support native developer role set `supports_developer_role = False`.
  - Example: `QwenLLMService`
  - Source: `pipecat-core-code/src/pipecat/services/qwen/llm.py:24-34`
- Reference examples commonly seed `developer` messages in context.
  - Example: `realtime-openai.py`
  - Source: `pipecat-core-code/examples/realtime/realtime-openai.py:180-186`
- The current Nemotron service locally rewrites `developer` to `system`.
  - Source: `src/nemotron_voice/services/nvidia/nemotron_omni.py:484-490`

### Consequence

As long as the Nemotron plan preserves `developer -> system`, it is not fully aligned with the `OpenAILLMService` / OpenAI-compatible adapter boundary. It remains a deliberate divergence in provider-visible prompt shape.

### Required Plan Constraints

- The plan should stop deferring this as a later phase if the stated goal is OpenAI/Pipecat alignment.
- It should normalize messages through the OpenAI adapter boundary and adopt one explicit policy:
  - `supports_developer_role = True` and pass native `developer` rows through
  - or `supports_developer_role = False` and use adapter-driven `developer -> user`
- It should not keep a local `developer -> system` rewrite if alignment is the goal.
- Given the current local Nemotron chat path, the closest OpenAI-compatible precedent is likely:
  - treat the service as not supporting native developer role
  - use adapter-driven `developer -> user`
  - keep service-level system instruction injection separate through the normal system-instruction path

## Synthesis Constraints for the Next `PLAN.md` Revision

The plan update should not patch these defects independently. The revised design needs a single coherent statement for all four:

- request snapshots are per HTTP pass, not the lasting source of truth for tool followups
- live `FunctionCallParams.context` behavior is preserved
- stock assistant aggregation remains the default for text-only behavior
- Nemotron-specific assistant/tool exactness is narrowly scoped to sync-tool turns
- queued stale-tool invalidation preserves both cancellation frames and cancellation events
- developer-role handling is settled now at the adapter boundary, not deferred
