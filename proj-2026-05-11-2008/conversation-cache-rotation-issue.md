# Conversation-cache `conversation_id` rotation on interrupted tool turns

**Status:** known, graceful degradation (not a correctness bug). A clean fix needs a
coordinated client + vLLM-engine change — see "Why the obvious fix didn't work".

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
just sent** (`src/nemotron_voice/services/nvidia/nemotron_omni.py:1441`):

```python
# _process_context, after a successful response:
self.committed_messages = copy.deepcopy(full_messages)
```

The next request is allowed to reuse the cache only if `committed_messages` is a
**prefix** of the new full transcript
(`src/nemotron_voice/services/nvidia/nemotron_omni.py:1039`):

```python
committed_prefix_matches = current_full[: len(self.committed_messages)] == self.committed_messages
...
non_append_rewrite = self._conversation_cache_committed and not committed_prefix_matches
if fingerprint_changed or non_append_rewrite:
    ...
    self._rotate_conversation_cache_projection(reason="non-append committed-prefix rewrite")
```

The bug: a **tool re-entry request** carries provisional rows that
`NemotronAssistantAggregator` may later **remove** from the live transcript
(when the bot is interrupted mid-turn). If the re-entry's response completes
*before* the interruption lands, the `committed_messages` snapshot has already
captured those provisional rows — and once the aggregator removes them, the live
transcript is no longer a prefix-extension of `committed_messages`. `_build_payload`
correctly detects that and rotates.

So: `committed_messages` is being anchored to rows that are **not stable** (they
can disappear), instead of to rows that *are* stable (the `user` rows).

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
pytest test that reproduces the rotation **deterministically** — it sets up the
state described above (a `committed_messages` snapshot that includes the
provisional tool rows) and then drives the post-interruption transcript through
`_build_payload`, the same code path the live bot uses.

Run it:

```bash
PYTHONPATH=src .venv-pipecat/bin/python -m pytest \
  proj-2026-05-11-2008/repro_conversation_cache_rotation.py -s -v
```

(or, since it is plain `pytest`, drop it into `tests/` and run it there.)

The test:

1. Builds a `NemotronOmniAudioLLMService(conversation_id="repro-conv")`.
2. Puts the service into the state it would be in *right after step (4) above*:
   `service._conversation_cache_committed = True` and
   `service.committed_messages = [system, user("Run pwd"), assistant("", tool_calls=[run_bash]), tool(result)]`
   — i.e. the committed mirror includes the two provisional rows. (It also copies
   the request's `cache_shape_fingerprint` onto `service._committed_cache_shape_fingerprint`
   so the only thing that can trigger a rotation is the prefix mismatch, not a
   shape change.)
3. Builds the *post-interruption* `LLMContext`: `[user("Run pwd"), user("What did the command print?")]`
   — the `assistant(tool_calls)` and `tool` rows were dropped by
   `NemotronAssistantAggregator._drop_all_provisional_sync_tool_rows`, and the new
   user turn was appended.
4. Calls `service._build_payload(service._normalized_request_snapshot(context))`
   (the exact call `_process_context` makes) and asserts:
   * `service._conversation_id` changed (a rotation occurred), and
   * the resulting payload has **no** `conversation_require_cache` (it went out as
     a full-history rebase).
   It also prints the `committed-prefix diverged` diagnostic so you can see the
   same field dump the live bot logs.

Step 2 is the only "synthetic" part; everything from step 3 on is real production
code. Steps 1–4 of "how a rotation happens" above are what produces that
`committed_messages` value in a live session — the repro skips replaying the tool
round (which needs the full aggregator + a registered tool handler) and just
asserts the value it lands on, then exercises the divergence path verbatim.

---

## Why the obvious fix didn't work (and what a real fix needs)

The natural fix is to anchor the cache-commit boundary at the **last `user`
row** — user rows are the one row class the assistant aggregator never rewrites
or removes after the fact, so `committed_messages` stays append-only across tool
turns. Concretely: in `_process_context`, set
`committed_messages = full_messages[: last_user_index + 1]`; and on the vLLM side,
truncate `committed_messages_after_success` the same way before publishing the
checkpoint (`render(messages[:last_user+1], gen_prompt=False)` is still a token
prefix of the full rendered prompt, so attach validation is unaffected). The two
mirrors must move together because the server reconstructs each suffix request by
prepending its own `committed_messages` — if the client truncates and the server
does not, every following turn 409s and rebases.

This was prototyped (all 66 service unit tests + 56 vLLM-side unit tests passed)
but **had to be reverted**: with the truncation, the committed boundary does *not*
advance during tool re-entries (it stays pinned at the same user row), so the
vLLM publish path is asked to "publish a checkpoint at a token position it
already has a checkpoint for". The conversation-cache scheduler doesn't handle
that gracefully — the engine wedges (`Running: 0 reqs, Waiting: 1 req`, EngineCore
spinning at ~80% CPU, no further progress) and even a trivial follow-up request
times out.

A real fix therefore needs an engine-aware change, e.g. one or more of:

* In `_publish_conversation_response` (and the engine commit), **skip the publish**
  when the new `committed_checkpoint_token_count` equals the existing checkpoint's
  token count (and the messages are identical) — i.e. don't re-publish a checkpoint
  that wouldn't move.
* Make the conversation-cache scheduler tolerate a "publish at an already-cached
  position" as a no-op (the block-copy path appears to be where it hangs —
  `Attached conversation cache ... copies=N`).
* Audit how `mamba_cache_mode='align'` snapshots the "terminal-state checkpoint"
  for a position in the *middle* of the rendered prompt (the truncated boundary is
  always mid-prompt; the current design's boundary — the full request prefix minus
  the generation prompt — is at the very end of the prompt).

Until then, the rotation stays as graceful degradation: correctness is preserved,
and the cost is one full-history re-render per occurrence (~1–2 per 20-turn
tool-heavy run). The `committed-prefix diverged` diagnostic remains in place so any
recurrence is self-explanatory in the logs.
