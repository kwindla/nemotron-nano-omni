# Suffix-Only Conversation Cache Plan

## Progress

- 2026-04-30: Implemented suffix-only append support.
  - The OpenAI frontend conversation cache now stores committed message
    snapshots and rendered prompt-token ledgers.
  - Same-`conversation_id` requests are locked before rendering, suffix messages
    are selected from full-history or suffix-only payloads, and rendering uses
    `committed_messages + suffix_messages`.
  - Publish now appends only suffix messages plus the generated assistant text
    to the frontend ledger after the engine pending cache publishes.
  - Rendered multimodal inputs are filtered so committed-prefix media is not
    sent to the engine encoder path; suffix media keeps absolute prompt
    positions.
  - The scheduler now warns when a committed engine state exists but the prompt
    is not longer than the committed token count, which indicates frontend
    reconstruction failed.
  - `NemotronOmniAudioLLMService` now defaults to suffix-only conversation
    payloads after the first successful cached turn, with a bot env toggle
    `NEMOTRON_OMNI_SUFFIX_ONLY_CONVERSATION=0` to disable it.
  - The managed endpoint harness now includes text suffix-only equivalence and
    audio suffix-only no-old-media-resend coverage.

## Goal

Support the local Pipecat voice-loop contract:

```text
conversation_id + latest user audio/message only
```

For a committed conversation, vLLM should internally prepend the cached
conversation head, render a correct full token row for generation, attach the
committed KV/Mamba state, and process only the new suffix multimodal inputs.

This is the missing phase after the current implementation. The tests that
already passed validate full-history requests with `conversation_id`; they do
not validate suffix-only append requests.

## Back-References

Primary context:

- Broader vLLM cache design and implementation history:
  `proj-vllm-nemotron-omni-conversation-prefix-cache.md`.
- Local voice-stack runbook and bot stack notes:
  `proj-local-nemotron-voice-stack.md`.
- Last managed cache integration result artifact:
  `conversation-prefix-cache-results.json`.
- Current managed vLLM integration harness:
  `scripts/test_vllm_conversation_prefix_cache.py`.
- Bot smoke harness for Pipecat/WebRTC/STT/TTS wiring:
  `scripts/smoke_step1_asr_bot.py`.

Pipecat code references:

- Current bot:
  `pipecat-core-code/examples/function-calling/function-calling-nemotron-omni-audio.py`.
- Saved known-working bot reference:
  `pipecat-core-code/examples/function-calling/function-calling-nemotron-omni-audio-working-reference-20260430.py`.
- Earlier context-focused reference:
  `pipecat-core-code/examples/function-calling/function-calling-nemotron-omni-audio-context-reference.py`.
- Nemotron Omni Pipecat service:
  `pipecat-core-code/src/pipecat/services/nvidia/nemotron_omni.py`.

vLLM source references:

- OpenAI chat protocol:
  `vllm-v0.20.0/vllm/entrypoints/openai/chat_completion/protocol.py`.
- OpenAI chat serving and stream publish/discard:
  `vllm-v0.20.0/vllm/entrypoints/openai/chat_completion/serving.py`.
- Frontend conversation cache ledger:
  `vllm-v0.20.0/vllm/entrypoints/openai/conversation_cache.py`.
- Render/preprocess path:
  `vllm-v0.20.0/vllm/entrypoints/serve/render/serving.py`.
- Engine conversation cache state:
  `vllm-v0.20.0/vllm/v1/core/conversation_cache.py`.
- Scheduler attach/stage/publish integration:
  `vllm-v0.20.0/vllm/v1/core/sched/scheduler.py`.
- Request token and multimodal feature bookkeeping:
  `vllm-v0.20.0/vllm/v1/request.py`.
- Multimodal placeholder/feature structures:
  `vllm-v0.20.0/vllm/multimodal/inputs.py`.

## Current State

- `conversation_id` exists on chat-completion requests.
- The OpenAI layer has same-id locking and publish/discard lifecycle.
- The engine can stage, publish, discard, and attach committed full-attention
  KV blocks plus terminal Mamba state.
- The scheduler sets `request.num_computed_tokens = state.token_count` when it
  attaches committed state.
- Pipecat currently sends the full `LLMContext` to vLLM, including prior audio
  turns.

Current limitation:

- vLLM only attaches the cache when the current rendered prompt already contains
  `committed prefix + new suffix`.
- If the client sends only the latest user audio/message, `state.token_count >=
  request.num_tokens`, so attach is skipped and the request is not a valid
  continuation of the committed head.

## API Contract

Add explicit append semantics for `conversation_id`:

1. First request for an id is a normal full request and creates the committed
   head after the assistant response completes.
2. Later requests for the same id may send only new messages.
3. The server trusts the client to send append-only suffix messages for that id.
4. The server combines the committed frontend message ledger with the request
   suffix messages before chat-template rendering.
5. The server publishes the assistant response by appending it to the frontend
   ledger and publishing the pending engine state.

Keep full-history compatibility:

- If a later request includes the full history, detect and drop the already
  committed prefix messages using the frontend committed message count.
- If a later request includes only suffix messages, use them as-is.
- Do not do expensive equality validation on the hot path.

## Frontend Ledger

Extend `vllm/entrypoints/openai/conversation_cache.py`.

Add to `ConversationFrontendEntry`:

- `committed_messages: list[ChatCompletionMessageParam]`
- `committed_prompt_token_ids: tuple[int, ...] | None`

On acquire:

- Return the existing entry, not only a lease, or add lookup helpers so
  `OpenAIServingChat` can obtain committed message count and messages after
  acquiring the lock.

On publish:

- Store the fully committed messages:
  - prior committed messages;
  - request suffix messages;
  - generated assistant message.
- Store committed prompt token ids if available from the rendered request or
  final response metadata.
- Continue storing `committed_message_count` and `committed_text` for debugging.

## Request Preparation

Modify `OpenAIServingChat.create_chat_completion()` in
`vllm/entrypoints/openai/chat_completion/serving.py`.

High-level flow:

1. Validate request.
2. Acquire conversation lease before rendering when `conversation_id` is set.
3. Get the frontend committed entry for this id.
4. Compute suffix messages:
   - If no committed messages exist: suffix is `request.messages`.
   - If `len(request.messages) > committed_message_count` and the request looks
     like a full-history request: suffix is
     `request.messages[committed_message_count:]`.
   - Otherwise suffix is `request.messages`.
5. Render `committed_messages + suffix_messages`.
6. Preserve `suffix_messages` on the lease/finalization path so publish appends
   only the new user/tool/developer messages plus the assistant response.

Important:

- The rendered prompt passed to the engine is still the full token row. This is
  required for token positions, stop checks, logits processors, output
  bookkeeping, and eventual committed token capture.
- The client no longer needs to upload old audio because old audio messages are
  reconstructed from the frontend ledger.

## Multimodal Handling

MVP for local single-process serving:

- Store committed messages in the frontend ledger exactly as received.
- For old audio turns, avoid re-fetching/re-decoding by storing a server-local
  URL or data URL that remains available for the session.
- Render full messages from the ledger plus suffix.
- After rendering, filter multimodal features before engine processing:
  - Prefix features satisfy
    `feature.mm_position.offset + feature.mm_position.length <= state.token_count`.
  - Suffix features satisfy
    `feature.mm_position.offset >= state.token_count`.
  - Any feature spanning the boundary is a 400 error.
  - Prefix features must not be scheduled for encoder processing.

Preferred implementation point:

- After `render_chat_request()` returns `engine_inputs`, but before
  `engine_client.generate()`, if a committed engine state exists for the
  conversation, remove prefix `mm_features` from the `EngineInput`.
- Keep `prompt_token_ids` as the full token row.
- Make sure suffix feature positions remain absolute positions in the full
  prompt.

Longer-term improvement:

- Store tokenized placeholder spans and multimodal UUID metadata in the frontend
  ledger so prefix placeholders can be reconstructed without old media payloads.
- For now, because this is a local process and short voice-loop use case, keeping
  committed audio data URLs in the frontend ledger is acceptable if old media is
  not resent by the client.

## Engine Attach Changes

Minimal engine change:

- The scheduler already attaches committed state when
  `state.token_count < request.num_tokens`.
- With frontend ledger reconstruction, suffix-only client requests become full
  token rows before reaching the engine, so this condition works.

Add safety:

- If `conversation_id` is set and a committed engine state exists, but attach is
  skipped because `state.token_count >= request.num_tokens`, return an error or
  log a hard warning. In suffix-only mode this indicates frontend reconstruction
  failed.

## Pipecat Changes

Modify `NemotronOmniAudioLLMService`:

- Add `suffix_only_conversation: bool = True` or similarly named setting.
- Track the latest audio user message created by `UserAudioContextCollector`.
- When `conversation_id` is set and suffix-only mode is enabled:
  - send only the latest user audio message in `messages`;
  - include the same `conversation_id`;
  - do not send prior user audio or assistant messages.
- Continue emitting LLM frames and assistant text frames as now.

Modify the bot pipeline:

- Keep STT branch for RTVI transcription events.
- Do not let STT final transcript become the LLM user message for the Omni LLM
  path.
- Use the audio collector as the user-turn source for the LLM path.
- Assistant aggregator can still maintain Pipecat-visible context for logging,
  but it should not force full-history payloads when suffix-only mode is active.

## Tests

Add endpoint/integration tests to
`scripts/test_vllm_conversation_prefix_cache.py`.

Required tests:

1. Text suffix-only equivalence:
   - First request sends a fact-setting user message with `conversation_id`.
   - Second request sends only the follow-up user question with the same id.
   - Compare with uncached full-history output under deterministic sampling.
   - Assert vLLM log contains `Attached conversation cache`.

2. Audio suffix-only correctness:
   - First request sends audio asking for a unicorn story.
   - Second request sends only a text follow-up asking what creature was
     mentioned.
   - Assert output contains `unicorn`.
   - Assert cache attach occurred.

3. Audio suffix-only no old-media resend:
   - First request sends audio.
   - Second request payload contains no old audio content.
   - Assert the second request still succeeds and attaches cache.
   - Inspect logs or instrumentation to confirm only suffix media is processed.

4. Full-history backward compatibility:
   - First request commits.
   - Second request sends full history.
   - Server drops committed prefix for ledger publishing but renders correct
     full prompt and attaches cache.

5. Pipecat bot smoke:
   - Run local vLLM as managed subprocess.
   - Run bot.
   - Send one audio turn, then a second audio/text turn through the aiortc
     client path.
   - Assert bot log shows latest-audio-only payload and vLLM attach on turn 2.

## Failure Cases

- Same id concurrent request returns 409.
- Malformed suffix after committed assistant state returns a normal model
  response only if the chat template accepts it; otherwise return 400.
- Prefix/suffix multimodal placeholder overlap returns 400.
- Client disconnect before `[DONE]` discards pending engine state and leaves the
  previous committed ledger unchanged.
- If frontend publish succeeds but engine publish fails, mark the frontend entry
  failed or discard the attempted append, preserving old committed messages.

## Implementation Order

1. Extend frontend cache entry to store committed messages.
2. Acquire lease before rendering and reconstruct messages for
   `conversation_id` requests.
3. Publish assistant response by appending suffix messages plus assistant text.
4. Add suffix-only text tests.
5. Filter prefix multimodal features and add suffix-only audio tests.
6. Update Pipecat service to send latest user audio only.
7. Run managed endpoint tests and bot smoke.
8. Update the main cache plan with completed status and measured results.

## Acceptance Criteria

- A same-`conversation_id` second request with only a new text message uses
  prior-turn context and attaches cache.
- A same-`conversation_id` second request with only new audio uses prior-turn
  context and attaches cache.
- Pipecat no longer uploads old audio turns to vLLM in suffix-only mode.
- STT transcripts still reach RTVI/client observers.
- Cached suffix-only output matches uncached full-history output for deterministic
  substantial prompts.
