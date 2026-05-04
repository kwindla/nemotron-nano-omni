## Conversation Cache Control Flow

This repo relies on three layers staying in sync:

1. Pipecat builds the shared user/assistant message history.
2. The local Nemotron service turns that history into either a full request or a
   suffix-only request.
3. vLLM reconstructs canonical history from its frontend ledger, renders the
   real prompt, uses that full rendered prompt as the prompt-only checkpoint
   boundary, then attaches the engine KV/Mamba cache only for an exact
   checkpoint prefix.

If any layer diverges, we get one of the failure signatures we have seen in
logs:

- `409 Conflict` / `already generating`
- `ConversationCacheMissError`
- `attach skipped`
- stale replies where a typed turn accidentally replays the last audio turn

### Audio Turn Path

```text
SmallWebRTC input audio
  -> LLMUserAggregator user-turn controller
  -> UserAudioContextCollector buffers raw audio for one turn
  -> UserStoppedSpeakingFrame
  -> UserAudioContextCollector appends exactly one multimodal user message
  -> AudioOnlyLLMUserAggregator.push_context_frame()
  -> NemotronOmniAudioLLMService receives LLMContextFrame
  -> suffix-only payload builder picks the latest user message
  -> vLLM frontend replays committed messages + new suffix message
  -> vLLM engine attaches exact prompt-checkpoint blocks
  -> assistant response is published back into the frontend ledger
  -> engine publish advances the durable checkpoint only if materialized
```

Invariant:

- One spoken turn must become exactly one `user` message in shared LLM context.
- That message must contain the audio part and must be the newest user turn.

### RTVI Text Path

```text
RTVI send-text
  -> LLMMessagesAppendFrame
  -> AudioOnlyLLMUserAggregator.add_messages()
  -> AudioOnlyLLMUserAggregator.push_context_frame()
  -> NemotronOmniAudioLLMService receives LLMContextFrame
  -> suffix-only payload builder picks the newest text user message
  -> vLLM frontend replays committed messages + text suffix
  -> engine attaches exact prompt-checkpoint blocks
```

Invariant:

- A typed turn after an audio turn must send the newest text message, not the
  previous audio user message.
- In logs, typed suffix-only turns should show `audio_parts=0`.

### Tool Call Followup Path

```text
top-level request
  -> model emits assistant tool call
  -> NemotronOmniAudioLLMService executes bash tool
  -> followup request sends only tool-result suffix
  -> tools/tool_choice stay present so chat-template rendering is stable
  -> vLLM publishes frontend history before yielding final [DONE]
  -> followup acquires the same conversation lease without 409
  -> engine attaches exact prompt-checkpoint blocks for the tool round
```

Invariants:

- Tool followup requests must keep `tools` and `tool_choice`.
- The followup request must arrive after the previous lease has been published
  and released.
- Assistant tool-call output is no longer committed as raw engine tail state.
  The frontend ledger replays it through the active chat template on the next
  request, and the engine commits only the rendered prompt checkpoint from
  before completion tokens are generated.
- Sequential tool rounds should never see `already generating` or `attach
  skipped`.

### Frontend Ledger Contract

For any request with `conversation_id`:

1. The frontend lease is acquired before rendering.
2. If the client sends suffix-only messages, the frontend ledger reconstructs
   the exact full prompt by prepending `lease.committed_messages`.
3. The caller-provided `cache_salt` is not the engine key directly. vLLM folds
   template, tools, and system/developer prompt inputs into an internal
   effective salt. Suffix-only requests may omit the system/developer messages,
   so the frontend keeps an alias from the caller salt to the committed
   effective salt and uses that alias only for suffix-only lookups.
4. The frontend renders the actual prompt with `add_generation_prompt=true` and
   uses those exact prompt token ids as the checkpoint candidate. For the current
   Nemotron chat template, that full prompt is a token prefix of the next
   canonical prompt after the assistant/tool result is replayed through the
   frontend ledger.
5. On the next request, the rendered prompt must exactly match or extend the
   committed prompt checkpoint. In `conversation_require_cache=true` mode, a
   mismatch returns `409 ConversationCacheMissError` before generation.
6. On success, the engine cache publish reports whether a physical checkpoint
   advanced. The frontend commits prompt token ids only to that engine-reported
   checkpoint length; otherwise it advances messages while preserving the last
   useful prompt checkpoint.

This is why `publish-before-[DONE]` matters for streaming responses: a client
that immediately sends the tool-result followup after the final SSE chunk should
still find the conversation lease ready.

### Engine Attach Contract

The engine will attach the conversation cache only if the rendered request
exactly matches or extends the committed prompt checkpoint. The important
checks are:

- committed engine state and physical blocks must exist
- checkpoint length must be non-zero
- `request_prompt_token_ids[:state.token_count] == state.committed_token_ids`
- request prompt length must be at least the checkpoint length
- Mamba terminal refs must be the exact saved refs for that checkpoint
- multimodal payloads that are fully inside the committed checkpoint are sent
  as cache-hit sentinels (`None`) while their placeholder descriptors remain in
  the request; this avoids reintroducing stale prefix audio/image work after an
  engine attach

Conversation cache must be operationally independent from vLLM's native
block-prefix-cache lookup. Native prefix caching may be disabled for benchmarking
or production policy, but Mamba `align` state layout must remain available so
conversation-owned terminal checkpoints can still be materialized and attached.

On the RTX 5090 NVFP4 profile, exact cached-vs-uncached output parity also
depends on preserving Mamba terminal state in `float32`. Lower-precision Mamba
cache state can still attach successfully, but it produced greedy audio-output
drift in the direct parity suite.

There is no longest-prefix or partial attach fallback. If
`conversation_require_cache=true`, attach misses become `409
ConversationCacheMissError` rather than silent uncached continuation.

If the rendered request is shorter than, or divergent from, the committed
prefix, the engine logs `attach skipped`.

### Regression Expectations

For a healthy sequential mixed session:

- first top-level turn may be uncached
- every later suffix-only top-level turn should attach cache unless the prior
  turn explicitly published a no-op checkpoint
- every tool followup should attach cache or return a typed 409 that triggers
  the full-context retry path
- no `409`, `ConversationCacheMissError`, or `attach skipped`
- typed followups after audio turns should answer the typed prompt, not replay
  the prior audio turn
