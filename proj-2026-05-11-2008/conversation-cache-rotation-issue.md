# Conversation-cache `conversation_id` rotation on interrupted tool turns

**Status:** fixed in this workspace. The clean fix was a coordinated client +
vLLM-serving + vLLM-scheduler change; see "What the fix does".

**Symptom:** during a long voice session, the bot occasionally logs

```
NemotronOmniAudioLLMService#0: rotating conversation_id from pipecat-XXX to pipecat-YYY after non-append committed-prefix rewrite
```

and the request that triggered the rotation goes out as a full-history rebase
(`conversation_require_cache` omitted, all messages re-sent) instead of a small
cache-reusing suffix. In a 20-turn mixed RTVI regression it happens ~1–2 times,
each costing one full-history re-render on the server. Output is unaffected — the
fresh `conversation_id` just re-establishes a checkpoint from scratch.

A diagnostic line is emitted right before each rotation so you can see exactly
what diverged:

```
NemotronOmniAudioLLMService#0: committed-prefix diverged before rotation:
  len(committed)=12 len(current_full)=12 first_diff_idx=9
  committed[9]={role='assistant' content='' tool_calls=['run_bash']}
  current_full[9]={role='user' content='What exact words did the previous spoken command print?...'}
```

---

## TL;DR of the root cause

The client (`NemotronOmniAudioLLMService`) keeps a mirror of "what messages the
conversation-cache checkpoint covers" in `self.committed_messages`. After every
successful request it snapshots that mirror from the **full transcript that was
just sent.

The next request is allowed to reuse the cache only if `committed_messages` is a
**prefix** of the new full transcript.

The first fix moved that durable boundary to the latest `user` row, which
solved the simple "re-entry transcript ends at `assistant(tool_calls)` +
`tool(result)`" case. The live bot still had one more edge: a newer **user**
row can already be present in the transcript while those earlier tool rows are
still provisional. In that shape, "latest user row" is still too optimistic,
because committing through that later user implicitly commits the provisional
tool rows that sit before it.

The real invariant is: the durable checkpoint may advance only through the
earliest user row whose following assistant/tool rows are still provisional.
The client therefore has to track unresolved sync-tool passes explicitly and use
that explicit stable boundary when it decides what `committed_messages` should
be after success.

---

## The components involved

* **`NemotronOmniAudioLLMService`** (`src/nemotron_voice/services/nvidia/nemotron_omni.py`)
  — the LLM client. Owns `committed_messages`, builds each request
  (`_build_payload`, line ~1026), runs it (`_process_context`, line ~1418),
  rotates on a non-append rewrite (`_rotate_conversation_cache_projection`, line
  ~1190). The pre-rotation diagnostic is `_describe_committed_prefix_divergence`
  (line ~1160).
* **`NemotronAssistantAggregator`** (`src/nemotron_voice/services/nvidia/nemotron_omni.py:254`)
  — the assistant-side context aggregator. For a sync tool turn it adds
  *provisional* rows to the `LLMContext` (`_handle_function_call_in_progress`,
  line ~331: `self._context.add_message(assistant_row)` for the
  `assistant(tool_calls)` row, then `self._context.add_message(tool_row)` for a
  placeholder `tool` row with `content="IN_PROGRESS"`). On `LLMFullResponseEndFrame`
  for the *follow-up* response those rows are "committed" (kept, stop being
  tracked as provisional — `_commit_response_candidates`, line ~480). On an
  `InterruptionFrame` they are instead **removed** (`_handle_interruptions` ->
  `_drop_all_provisional_sync_tool_rows` -> `_remove_provisional_rows`, lines
  ~441/487/500, which does `context.transform_messages(... drop those rows ...)`).
* **vLLM `OpenAIServingChat`** (`vllm-v0.20.0/vllm/entrypoints/openai/chat_completion/serving.py`)
  — the server side of the cache. After a generation it publishes a checkpoint
  for `committed_messages_after_success = copy.deepcopy(list(render_request.messages))`
  (line ~426) — i.e. it commits the **full** reconstructed request, mirroring the
  client's `committed_messages`. The checkpoint tokens are
  `render(committed_messages, gen_prompt=False)`
  (`_render_conversation_prompt_token_ids`, line ~982; published in
  `_publish_conversation_response`, line ~1218). Client and server therefore must
  agree on `committed_messages` exactly.
* **bot wiring** (`src/nemotron_voice/bot.py`) — picks a `conversation_id`
  (line ~503-506: `Using Nemotron Omni conversation_id=...`), constructs the
  service with it (line ~518), and wires the `InterruptedToolPassSignal` between
  the aggregator and the `UserAudioContextCollector` (line ~334, ~406).

---

## Step-by-step: how a rotation happens

Let `H` be the settled transcript before the tool turn, ending in a user row, e.g.
`H = [system, ..., assistant("..."), user("use bash and run echo two")]`.
Suppose `committed_messages == H` and the server's checkpoint covers `render(H)`.

1. **Tool-call request.** The client sends a tiny suffix relative to `H` (often
   just the new user row) with `conversation_require_cache=true`. The model
   replies with **`output_text="" , tool_calls=[run_bash]`** (a tool call, no
   spoken text). `_process_context` runs `self.committed_messages = full_messages`
   — `full_messages` here is still `H` (no tool rows in the transcript yet), so
   `committed_messages == H`. Then it emits a `NemotronExactAssistantMessageFrame`
   + `FunctionCallsStartedFrame`/`FunctionCallInProgressFrame`.

2. **Aggregator adds provisional rows.** `NemotronAssistantAggregator
   ._handle_function_call_in_progress` appends two rows to the live `LLMContext`:
   * `A = {role: "assistant", content: "", tool_calls: [run_bash(...)]}`  (line ~354)
   * `R = {role: "tool", content: "IN_PROGRESS", tool_call_id: ...}`        (line ~366)

   These are tracked in `provisional_rows` of a `_NemotronProvisionalSyncToolPass`.

3. **Tool runs; result fills in.** `_run_bash_tool` returns; the base aggregator
   updates `R["content"]` to the JSON result object. Live transcript is now
   `H + [A, R]`.

4. **Tool follow-up request (re-entry).** The bot re-enters: `_process_context`
   runs again. `full_messages = H + [A, R]`. The model replies with plain text
   (e.g. `"spark text two"`, `tool_calls=[]`). `_process_context` does
   `self.committed_messages = copy.deepcopy(full_messages)` -> **`committed_messages == H + [A, R]`**.
   On the server side the matching publish commits `render(H + [A, R])`.
   *(The follow-up's text is suppressed by the aggregator while a sync-tool pass
   is active — `_should_suppress_aggregation_commit`, lines ~455-459 — so no extra
   `assistant("spark text two")` row is appended.)*

5. **Interruption lands before the follow-up's `LLMFullResponseEndFrame` is
   processed by the aggregator.** In the real bot, frames between processors travel
   through async queues, so there is a window between "(4) `_process_context`
   returned" and "the aggregator processes the follow-up's
   `LLMFullResponseEndFrame`" (which is what calls `_commit_response_candidates`
   and would make `A`,`R` permanent). The mixed-RTVI regression sends the *next*
   audio turn right on the heels of a text-tool turn, so its `InterruptionFrame`
   often arrives in exactly that window. `_handle_interruptions` ->
   `_drop_all_provisional_sync_tool_rows()` -> `_remove_provisional_rows([A, R])`
   removes `A` and `R` from the live transcript. The follow-up's later
   `LLMFullResponseEndFrame` then sees `_current_response_interrupted == True` and
   skips `_commit_response_candidates`. Live transcript is back to `H`.

6. **The interrupting user turn arrives.** `UserAudioContextCollector` appends the
   new user row `U2` (with a `<user_interruption>` marker since the signal was
   set). Live transcript = `H + [U2]`.

7. **Next request -> rotation.** `_process_context` -> `_build_payload`:
   `current_full == H + [U2]`, but `committed_messages == H + [A, R]`.
   `current_full[: len(committed_messages)]` is `H + [U2]` truncated to
   `len(H)+2` items, i.e. `H + [U2, <whatever is at len(H)+1>]` — at index
   `len(H)` it has `U2`, while `committed_messages[len(H)]` is `A`. Not equal ->
   `committed_prefix_matches == False` -> `non_append_rewrite == True` ->
   `_rotate_conversation_cache_projection("non-append committed-prefix rewrite")`.
   The diagnostic prints `first_diff_idx = len(H)`, `committed[i] = A` (the
   `assistant(content="", tool_calls=[run_bash])` row), `current_full[i] = U2`.
   That is exactly the field dump quoted at the top of this doc.

Because step (5) is a race (the interruption has to land in that window), this is
intermittent — ~1–2 times per 20-turn run, and only on turns that immediately
follow a tool turn.

---

## Repro

`proj-2026-05-11-2008/repro_conversation_cache_rotation.py` is a self-contained
pytest regression that reconstructs the post-re-entry state under the fixed
contract and then proves the interrupted follow-up reuses the same
`conversation_id` with a one-message user suffix.

Run it:

```bash
PYTHONPATH=src .venv-pipecat/bin/python -m pytest \
  proj-2026-05-11-2008/repro_conversation_cache_rotation.py -s -v
```

(or, since it is plain `pytest`, drop it into `tests/` and run it there.)

The test:

1. Builds a `NemotronOmniAudioLLMService(conversation_id="repro-conv")`.
2. Reconstructs the tool re-entry transcript
   `[system, user("Run pwd"), assistant("", tool_calls=[run_bash]), tool(result)]`
   and runs it through the same "committable after success" rule the service now
   uses. The durable boundary truncates back to `[system, user("Run pwd")]`.
   (It also copies the request's `cache_shape_fingerprint` onto
   `service._committed_cache_shape_fingerprint` so the only thing under test is
   the committed-prefix behavior.)
3. Builds the *post-interruption* `LLMContext`: `[user("Run pwd"), user("What did the command print?")]`
   — the `assistant(tool_calls)` and `tool` rows were dropped by
   `NemotronAssistantAggregator._drop_all_provisional_sync_tool_rows`, and the new
   user turn was appended.
4. Calls `service._build_payload(service._normalized_request_snapshot(context))`
   (the exact call `_process_context` makes) and asserts:
   * `service._conversation_id` stays the same,
   * the resulting payload keeps `conversation_require_cache=true`, and
   * the payload suffix is just the new user row.

Step 2 is the only "synthetic" part; everything from step 3 on is real production
code. The repro still bypasses the live tool round, but it now verifies the new
stable-boundary rule directly and then exercises the exact `_build_payload`
branch that used to rotate.

---

## What the fix does

The durable cache boundary is now **client-authoritative**:

* `NemotronOmniAudioLLMService` tracks unresolved sync-tool passes explicitly.
  While any provisional pass is still live, the durable boundary stays pinned at
  that pass's anchor `user` row instead of advancing through later user rows
  that merely happen to be present in the transcript.
* The client sends vLLM the exact
  `conversation_committed_message_count` it wants published after a successful
  pass, so vLLM no longer has to infer the durable boundary from "latest user
  row" heuristics.
* `NemotronAssistantAggregator` clears those provisional-batch markers only when
  the pass is truly committed (`LLMFullResponseEndFrame` with no interruption)
  or explicitly dropped on interruption/cancellation.

The earlier same-boundary vLLM fixes are still required and still present:

* `_publish_conversation_response` accepts an `advanced_checkpoint=false`
  publish result as success when it exactly reuses the already-committed
  checkpoint length.
* `Scheduler._cap_conversation_checkpoint_schedule()` no longer returns `0`
  when the logical boundary is already fully attached/computed; it keeps
  scheduling the request normally and lets the later publish no-op against the
  existing checkpoint.

Together, that means interrupted tool turns stay append-only from the durable
boundary's point of view even when a newer user row is already in the context,
so the bot stops rotating `conversation_id` on this path and same-boundary tool
re-entries still avoid wedging the engine.
