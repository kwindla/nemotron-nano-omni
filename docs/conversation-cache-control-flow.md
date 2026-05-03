## Conversation Cache Control Flow

This repo relies on three layers staying in sync:

1. Pipecat builds the shared user/assistant message history.
2. The local Nemotron service turns that history into either a full request or a
   suffix-only request.
3. vLLM reconstructs the exact prompt prefix from its frontend ledger, then
   attaches the engine KV/Mamba cache for the replayable prefix.

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
  -> vLLM engine attaches replayable prefix blocks
  -> assistant response is published back into frontend + engine ledgers
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
  -> engine attaches replayable prefix blocks
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
  -> vLLM publishes conversation state before yielding final [DONE]
  -> followup acquires the same conversation lease without 409
  -> engine attaches replayable prefix blocks for the tool round
```

Invariants:

- Tool followup requests must keep `tools` and `tool_choice`.
- The followup request must arrive after the previous lease has been published
  and released.
- Sequential tool rounds should never see `already generating` or `attach
  skipped`.

### Frontend Ledger Contract

For any request with `conversation_id`:

1. The frontend lease is acquired before rendering.
2. If the client sends suffix-only messages, the frontend ledger reconstructs
   the exact full prompt by prepending `lease.committed_messages`.
3. The rendered prompt token ids are recorded as the committed prefix for the
   next turn.
4. On success, the engine cache is published first, then the frontend ledger is
   published, then the lease is released.

This is why `publish-before-[DONE]` matters for streaming responses: a client
that immediately sends the tool-result followup after the final SSE chunk should
still find the conversation lease ready.

### Engine Attach Contract

The engine will attach the conversation cache only if the rendered request
extends the replayable committed prefix. The important checks are:

- shared prefix token count must be positive
- attachable token count must be less than request token count
- Mamba align mode may trim the replayable prefix to the last compatible block
  boundary

If the rendered request is shorter than, or divergent from, the committed
prefix, the engine logs `attach skipped`.

### Regression Expectations

For a healthy sequential mixed session:

- first top-level turn may be uncached
- every later suffix-only top-level turn should attach cache
- every tool followup should attach cache
- no `409`, `ConversationCacheMissError`, or `attach skipped`
- typed followups after audio turns should answer the typed prompt, not replay
  the prior audio turn
