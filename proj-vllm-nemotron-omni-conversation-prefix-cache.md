# vLLM Nemotron Omni Exact Conversation Cache Plan

Last reviewed against local checkout `vllm-v0.20.0` at git commit
`88d34c6`.

## Progress

- 2026-04-30: Phase 1 schema/plumbing gate complete.
  - Added top-level `conversation_id` to vLLM chat-completion requests.
  - Threaded the id through render prompt extras, engine inputs,
    `EngineCoreRequest`, and v1 `Request`.
  - Added Pipecat bot/service plumbing so each bot session sends one stable
    `conversation_id` on every Nemotron Omni audio-LLM request.
  - Validated with in-process vLLM/Pipecat smoke checks, a live text request to
    `/v1/chat/completions`, and an aiortc audio bot smoke through
    SmallWebRTCTransport.
- 2026-04-30: Phase 2 OpenAI frontend ledger/lease gate complete.
  - Added a process-local conversation frontend cache manager that keys by
    `(model, cache_salt, conversation_id)`, rejects concurrent same-id
    requests with `409 Conflict`, and records the committed assistant text and
    message count after successful completion.
  - Added streaming response wrapping so final `[DONE]` publishes the turn and
    stream close/disconnect discards the pending lease.
  - Added unsupported-option validation for `conversation_id` requests:
    `n > 1`, beam search, `continue_final_message`, and
    `add_generation_prompt=false`.
  - Validated with syntax/diff checks, an in-process cache smoke, sequential
    live same-id requests, concurrent busy conflict, and interrupted-stream
    cleanup through `/v1/chat/completions`.
  - Re-ran the aiortc Pipecat bot smoke on a separate test port with Smart Turn
    and local Pocket TTS against the patched vLLM endpoint.
  - This phase does not reuse KV or Mamba state yet. It establishes the
    request lifecycle and locking contract that engine-side cache attachment
    will depend on.
- 2026-04-30: Phase 3 engine control-plane state gate complete.
  - Added an engine-side exact conversation cache manager that records pending
    and committed complete-turn token ledgers keyed by
    `(conversation_id, cache_salt)`.
  - Added `conversation_generation_id` request plumbing and scheduler staging
    before request metadata teardown.
  - Added publish/discard/touch/delete utility methods through `EngineClient`,
    `AsyncLLM`, `CoreClient`, `EngineCore`, and the scheduler.
  - Wired OpenAI streaming and non-streaming success paths to publish pending
    engine state, and failure/disconnect paths to discard pending state.
  - Validated with syntax/diff checks, an in-process engine-cache smoke, live
    sequential same-id requests, concurrent busy conflict, interrupted-stream
    cleanup, vLLM log inspection for cache warnings, and the aiortc Pipecat bot
    smoke on a separate test port.
  - This phase still does not attach, pin, copy, or reuse KV/Mamba state. It
    establishes the engine state and control methods that the physical cache
    phases will use.
- 2026-04-30: Phase 4a full-attention block ownership primitives complete.
  - Added KV manager helpers to read request block refs, remove request
    bookkeeping without freeing refs, attach existing refs to another request,
    and release conversation-owned refs.
  - Added a focused full-attention refcount smoke that verifies the ownership
    transfer shape: request refs become conversation-owned refs, attach bumps
    refs for a new request, conversation release drops only its ownership, and
    request free releases the remaining refs.
  - Restarted vLLM with these helpers and re-ran a `conversation_id` text smoke
    plus the aiortc Pipecat bot smoke.
  - These helpers are not yet wired into scheduler attach/commit for Nemotron
    Omni. The hybrid model path must stay inactive until terminal Mamba
    attach/capture is implemented.
- 2026-04-30: Phase 5a inactive Mamba terminal-state primitives complete.
  - Added explicit terminal Mamba state reference/capture data structures for
    exact complete-turn caching.
  - Added Mamba align-mode helpers to compute the terminal state block-table
    index `(token_count - 1) // block_size`, read the terminal state block ref,
    attach an existing terminal state ref to a new request at the same logical
    index, and release conversation-owned terminal refs.
  - Added KV manager wrappers and a focused Mamba refcount smoke that verifies
    null padding, terminal block position, attach ref increments, and release
    semantics without using vLLM's Mamba prefix hash matching path.
  - Restarted vLLM and re-ran a `conversation_id` text smoke plus the aiortc
    Pipecat bot smoke.
  - These helpers are still inactive. The next phase must add worker-side
    terminal Mamba capture/copy execution before scheduler attach can be
    enabled for Nemotron Omni.
- 2026-04-30: Phase 5b inactive worker-side Mamba terminal capture plumbing
  complete.
  - Added terminal capture metadata to `SchedulerOutput`.
  - Added a worker-side Mamba capture helper that uses the existing
    `collect_mamba_copy_meta()` and `do_mamba_copy_block()` path, scoped to
    explicit exact conversation capture records instead of Mamba prefix hash
    matching.
  - Added a `GPUModelRunner._update_states()` hook that runs captures before
    finished request state is removed. Because the scheduler does not emit
    capture records yet, this remains behaviorally inert.
  - Validated with syntax/diff checks, scheduler-output/capture-helper smoke,
    live `conversation_id` text smoke, vLLM log inspection, and the aiortc
    Pipecat bot smoke.
  - Next phase: have the scheduler allocate/request a pending terminal Mamba
    destination state block, emit capture metadata on successful conversation
    request finish, and keep the captured pending state pinned until frontend
    publish/discard.
- 2026-04-30: Phase 6a active pending physical block lifetime complete.
  - Extended engine conversation state to own physical KV/Mamba block refs by
    cache group while pending or committed.
  - The scheduler now stages a successfully finished conversation request by
    transferring current request block refs into pending engine state, then
    removing request block bookkeeping before normal request cleanup. This keeps
    worker request teardown intact without releasing conversation-owned blocks.
  - Publish, discard, delete, and pending-TTL paths now return old/new states
    that must be released; the scheduler drops their physical block refs through
    `KVCacheManager.release_block_refs_by_group()`.
  - Conversation physical staging is disabled when KV connector / P-D transfer
    paths are active, because those paths also own delayed block-free behavior.
  - Terminal Mamba metadata is collected only in `align` cache mode. The current
    local validation command does not pass `--mamba-cache-mode align`, so this
    gate validated physical block ownership, frontend/engine lifecycle, and
    rollback cleanup, but not terminal Mamba capture/reuse.
  - Validated with syntax/diff checks, an in-process active block-lifetime
    smoke, live sequential same-id requests, live busy/disconnect cleanup, vLLM
    log inspection, and the aiortc Pipecat bot smoke.
  - Attach/reuse remains disabled. The next phase is request attach with
    copy-on-write for full-attention tail blocks and request-owned Mamba
    terminal copies.
- 2026-04-30: Phase 6b full-attention copy-on-write attach plumbing complete.
  - Added `ConversationBlockCopy` scheduler metadata for exact conversation
    block copies.
  - Added full-attention attach helpers that share complete committed blocks by
    refcount and allocate a fresh request-owned copy for a non-block-aligned
    committed tail.
  - Added worker-side execution for full-attention conversation block copies.
    The worker expands logical vLLM blocks to kernel blocks when the attention
    backend uses a smaller kernel block size, then copies along the backend's
    KV block dimension before execution uses the block table.
  - Added scheduler attach plumbing for full-attention-only committed states.
    Hybrid Nemotron states are rejected before refs are touched, so the live
    Nemotron Omni path still stages and releases physical refs but does not
    reuse cached prefixes yet.
  - Validated with syntax/diff checks, focused full-attention COW and hybrid
    rejection smokes, live sequential same-id requests, live busy/disconnect
    cleanup, vLLM log inspection, and the aiortc Pipecat bot smoke.
  - Next phase: implement request-owned Mamba terminal attach/copy and terminal
    capture allocation so hybrid Nemotron states can safely enable reuse.
- 2026-04-30: Phase 6c Mamba terminal copy-on-write attach plumbing complete.
  - Added Mamba terminal attach helpers that allocate a request-owned terminal
    state block instead of pointing the request at the committed state block.
  - Added worker-side physical Mamba terminal block copies using the existing
    Mamba state-copy function and copy buffer path.
  - Added combined hybrid conversation attach plumbing: full-attention complete
    blocks are shared, full-attention partial tails are copied, and Mamba
    terminal states are copied into request-owned blocks.
  - The scheduler now reconstructs committed terminal Mamba refs from engine
    state metadata and can attach hybrid states when the server is running in
    `--mamba-cache-mode align`.
  - Validation showed this checkout forces Mamba cache mode to `none` unless
    `--enable-prefix-caching` is also set. With `--enable-prefix-caching
    --mamba-cache-mode align --mamba-backend triton`, local startup needed
    `--gpu-memory-utilization 0.75`; `0.72` failed the KV-capacity check while
    another GPU service was using about 3.2 GB.
  - Validated with syntax/diff checks, focused Mamba-only and hybrid COW attach
    smokes, a live append-style three-turn same-id text conversation under
    align mode, live busy/disconnect cleanup, vLLM log inspection, and the
    aiortc Pipecat bot smoke.
  - Added an INFO-level attach-hit log for the experimental path. A live
    append-style second turn confirmed: `Attached conversation cache ... tokens=26
    copies=5`.
  - Next phase: confirm terminal state correctness against an uncached baseline
    and add structured metrics before relying on TTFT measurements.
- 2026-04-30: Suffix-only conversation append phase implemented.
  - Added frontend committed-message snapshots and prompt-token ledgers so a
    later `conversation_id` request can send only new suffix messages.
  - OpenAI chat serving now acquires the same-id lease before rendering,
    reconstructs the full message row for tokenization, filters already-cached
    prefix multimodal inputs before engine generation, and publishes only the
    suffix plus generated assistant text.
  - Pipecat's Nemotron Omni service now defaults to latest-user-message payloads
    after the first successful cached turn, so old audio is not resent by the
    bot in suffix-only mode.
  - Added managed endpoint tests for text suffix-only equivalence and audio
    suffix-only follow-up/no-old-media-resend behavior.

## Goal

Patch vLLM so OpenAI-compatible chat completions can accept a
`conversation_id` and reuse a complete, append-only multimodal conversation
state for Nemotron 3 Nano Omni.

This is not generic vLLM prefix caching. The cache is valid only at complete
turn boundaries and only for exact append-only continuation of one committed
conversation head. There is no rewind, partial prefix matching, Mamba block
hash search, or arbitrary block-boundary reuse.

## Decision From Review

Use a new engine-level exact conversation cache.

The external review correctly identified existing vLLM infrastructure:

- `ChatCompletionRequest.cache_salt` already exists at
  `vllm/entrypoints/openai/chat_completion/protocol.py:323`.
- Multimodal hashes, salt, LoRA, and prompt-embed hashes are already mixed into
  normal block hashes in `vllm/v1/core/kv_cache_utils.py`.
- `MambaManager.find_longest_cache_hit()` already implements a Mamba block
  prefix-cache path in
  `vllm/v1/core/single_type_kv_cache_manager.py:806`.
- The Nemotron Omni wrapper delegates Mamba state shape/copy helpers, but does
  not declare `SupportsMambaPrefixCaching` at
  `vllm/model_executor/models/nano_nemotron_vl.py:901`.

Those findings change the plan, but they do not change the chosen mechanism.
For this local multi-turn voice use case we should not rely on vLLM's partial
Mamba block prefix cache. Instead, we store and reattach the exact full
conversation state at committed turn boundaries:

- full-attention KV blocks for every committed token;
- the exact terminal Mamba recurrent state for the committed token length;
- the committed prompt token ledger and multimodal placeholder/UUID metadata.

This avoids the hard cases in general Mamba caching: no longest-prefix search,
no sub-turn rewind, no partial Mamba block reuse, and no block-boundary
realignment decision. The cost is that we must write new session-cache code in
the scheduler, KV managers, worker copy path, and OpenAI serving layer.

Important distinction: vLLM still stores KV and Mamba state in block-addressed
GPU memory, because that is how the worker kernels and block tables address
cache tensors. Our new cache must not use vLLM's block-based Mamba prefix-cache
algorithm. Blocks are only the physical storage units for an exact turn state.
The Mamba semantic cache entry is one terminal recurrent state for the exact
committed token length.

## Scope

Supported in v1:

- `/v1/chat/completions` only.
- One served model, one local vLLM engine process.
- `stream=true` primary path; non-streaming handled but not optimized.
- `n == 1`.
- No beam search.
- No speculative decoding for the first implementation.
- `--max-num-seqs 1` recommended for the local voice loop.
- `--mamba-cache-mode=align` required for Nemotron Omni.
- `--enable-prefix-caching` may remain enabled, but conversation reuse does not
  depend on normal block-hash prefix hits.

Explicitly out of scope for v1:

- `/v1/completions`, `/v1/responses`, Realtime, batch serving.
- data-parallel or load-balanced multi-replica cache sharing;
- cross-process or persisted conversation cache;
- partial prefix validation against user-supplied history;
- rewind/edit/fork of a cached conversation;
- serving multiple same-`conversation_id` requests concurrently.

## API Contract

Add `conversation_id: str | None` to `ChatCompletionRequest` near the existing
extension fields in
`vllm/entrypoints/openai/chat_completion/protocol.py:150`.

For this local cache mode:

1. The first request for a `conversation_id` creates the committed head after
   the model response completes.
2. Later requests with the same id are treated as append-only continuations.
3. The client may send the full `messages` list, but the server trusts the
   client's cache state. It uses the stored committed message count to treat
   earlier messages as already cached and does not perform expensive equality
   validation.
4. Add an optional debug-only validation flag later:
   `--conversation-cache-validate-prefix`. That mode can re-render and compare
   token prefixes, but it is not on the hot path.

`cache_salt` remains orthogonal:

- The conversation cache key is `(served_model, cache_salt_or_default,
  conversation_id)`.
- If the request omits `cache_salt`, derive an internal salt from the model
  name and `conversation_id` so fallback normal prefix cache entries are not
  accidentally shared across conversations.
- If the request provides `cache_salt`, include it in the key and pass it
  through to existing vLLM hashing unchanged.

Reject with `400` when `conversation_id` is used with unsupported request
features. Return `409 Conflict` when the same id is busy.

## Source Map

OpenAI request/serving:

- `ChatCompletionRequest` is defined at
  `vllm/entrypoints/openai/chat_completion/protocol.py:150`.
- `messages` is at `protocol.py:153`.
- `cache_salt` is at `protocol.py:323`.
- `vllm_xargs` is at `protocol.py:340`; we should use a real top-level
  `conversation_id`, not hide this behind `vllm_xargs`.
- `OpenAIServingChat.create_chat_completion()` starts at
  `vllm/entrypoints/openai/chat_completion/serving.py:229`.
- The request is rendered at `serving.py:251`.
- `engine_client.generate()` is called at `serving.py:341`.
- The streaming generator starts at `serving.py:525`.
- Streaming consumes `RequestOutput` at `serving.py:635`.
- The stream emits `[DONE]` at `serving.py:1271`.
- The non-streaming generator starts at `serving.py:1273`.

Engine request plumbing:

- `EngineClient.generate()` is declared at `vllm/engine/protocol.py:65`.
- `AsyncLLM.generate()` starts at `vllm/v1/engine/async_llm.py:521`.
- `AsyncLLM.generate()` handles client disconnects at
  `async_llm.py:583` and calls `abort()` at `async_llm.py:588`.
- `EngineCoreRequest` starts at `vllm/v1/engine/__init__.py:80`.
- `EngineCoreRequest.cache_salt` is at `vllm/v1/engine/__init__.py:93`.
- `InputProcessor` builds `EngineCoreRequest` at
  `vllm/v1/engine/input_processor.py:360`.
- `Request.__init__()` starts at `vllm/v1/request.py:60`.
- `Request.num_computed_tokens` initializes at `vllm/v1/request.py:136`.
- `Request.from_engine_core_request()` starts at `vllm/v1/request.py:177`.
- `Request.append_output_token_ids()` updates all token ids at
  `vllm/v1/request.py:200`.

Scheduler/KV lifetime:

- `SchedulerOutput` is defined at `vllm/v1/core/sched/output.py:179`.
- `Scheduler` constructs `SchedulerOutput` at
  `vllm/v1/core/sched/scheduler.py:923`.
- Stopped requests are handled at `scheduler.py:1433`.
- Finished requests are freed at `scheduler.py:1441` before the OpenAI stream
  generator has finished yielding to HTTP.
- `_free_request()` starts at `scheduler.py:1826`.
- `_free_blocks()` calls `kv_cache_manager.free()` and deletes the request at
  `scheduler.py:1844`.
- `KVCacheManager.get_computed_blocks()` starts at
  `vllm/v1/core/kv_cache_manager.py:183`; it is normal block prefix caching
  and is not the conversation-cache reuse mechanism.
- `KVCacheManager.allocate_slots()` starts at
  `vllm/v1/core/kv_cache_manager.py:264`.
- `KVCacheManager.free()` starts at `kv_cache_manager.py:436`.
- `SingleTypeKVCacheManager.req_to_blocks` is defined at
  `vllm/v1/core/single_type_kv_cache_manager.py:73`.
- `SingleTypeKVCacheManager.num_cached_block` is defined at
  `single_type_kv_cache_manager.py:79`.
- `SingleTypeKVCacheManager.allocate_new_blocks()` starts at
  `single_type_kv_cache_manager.py:238`.
- `SingleTypeKVCacheManager.cache_blocks()` starts at
  `single_type_kv_cache_manager.py:273`.
- `SingleTypeKVCacheManager.free()` starts at
  `single_type_kv_cache_manager.py:299`.
- `BlockPool.get_new_blocks()` starts at `vllm/v1/core/block_pool.py:322`.
- `BlockPool.touch()` starts at `block_pool.py:391`.
- `BlockPool.free_blocks()` starts at `block_pool.py:408`.

Worker and block tables:

- `CachedRequestState` is defined at
  `vllm/v1/worker/gpu_input_batch.py:30`.
- `InputBatch.add_request()` starts at `gpu_input_batch.py:320`.
- `InputBatch.remove_request()` starts at `gpu_input_batch.py:489`.
- `GPUModelRunner._update_states()` starts at
  `vllm/v1/worker/gpu_model_runner.py:1061`.
- Finished request worker state is removed at `gpu_model_runner.py:1071`.
- New request worker state is built at `gpu_model_runner.py:1158`.
- Running request block tables are extended at `gpu_model_runner.py:1303`.
- `BlockTable.append_row()` starts at `vllm/v1/worker/block_table.py:102`.
- `MultiGroupBlockTable.add_row()` starts at
  `vllm/v1/worker/block_table.py:280`.

Mamba:

- `MambaManager` starts at
  `vllm/v1/core/single_type_kv_cache_manager.py:790`.
- `MambaManager.find_longest_cache_hit()` starts at
  `single_type_kv_cache_manager.py:806`; do not use this for conversation
  cache reuse.
- `MambaManager.allocate_new_blocks()` align-mode logic starts at
  `single_type_kv_cache_manager.py:954`.
- `MambaManager.free()` starts at `single_type_kv_cache_manager.py:1032`.
- `MambaManager.get_num_skipped_tokens()` documents that only the last
  computed-token state is needed at `single_type_kv_cache_manager.py:1038`.
- `mamba_utils.preprocess_mamba()` starts at
  `vllm/v1/worker/mamba_utils.py:147`.
- `mamba_utils.postprocess_mamba()` starts at `mamba_utils.py:222`; it only
  copies at aligned full-block boundaries and is not sufficient for arbitrary
  complete-turn capture.
- `mamba_get_block_table_tensor()` starts at
  `vllm/v1/attention/backends/utils.py:860`; align mode gathers the block at
  `(seq_len - 1) // block_size` at `utils.py:887`.
- Mamba attention computes `has_initial_states_p` from
  `num_computed_tokens > 0` starting at
  `vllm/v1/attention/backends/mamba_attn.py:439`.

Model/config:

- `mamba_cache_mode` is defined in `vllm/config/cache.py:132`.
- `NemotronHForCausalLM` declares `SupportsMambaPrefixCaching` at
  `vllm/model_executor/models/nemotron_h.py:765`.
- `NemotronH_Nano_VL_V2` starts at
  `vllm/model_executor/models/nano_nemotron_vl.py:901` and delegates Mamba
  state helpers at `nano_nemotron_vl.py:1587`.

External references:

- Nemotron 3 Omni report:
  https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Omni-report.pdf
- Nemotron 3 Nano report:
  https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf
- Mamba:
  https://arxiv.org/abs/2312.00752
- Mamba-2 / SSD:
  https://arxiv.org/abs/2405.21060
- Model card:
  https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4

## New Data Structures

Add an OpenAI-layer manager, for example
`vllm/entrypoints/openai/conversation_cache.py`:

```python
class ConversationCacheState(Enum):
    READY = "ready"
    GENERATING = "generating"
    ENGINE_COMMITTED = "engine_committed"
    FRONTEND_COMMITTING = "frontend_committing"
    FAILED = "failed"

@dataclass
class ConversationFrontendEntry:
    key: ConversationCacheKey
    state: ConversationCacheState
    lock: asyncio.Lock
    generation_id: str | None
    committed_message_count: int
    committed_prompt_token_ids: list[int]
    committed_text: str
    cache_salt: str | None
    media_ledger: ConversationMediaLedger
    created_at: float
    last_access: float
```

Add engine-core state, for example `vllm/v1/core/conversation_cache.py`:

```python
@dataclass
class ConversationEngineState:
    key: ConversationCacheKey
    generation_id: str
    token_count: int
    committed_token_ids: list[int]
    block_ids_by_group: tuple[list[int], ...]
    # Physical storage locations for exact terminal Mamba states. These are not
    # block-prefix-cache hits and are never matched by hash.
    mamba_terminal_state_blocks: dict[int, int]
    block_size_by_group: tuple[int, ...]
    cache_salt: str | None
    last_access: float
    bytes_estimate: int

@dataclass
class PendingConversationEngineState:
    previous_generation_id: str | None
    pending_generation_id: str
    old_state: ConversationEngineState | None
    new_state: ConversationEngineState
```

The frontend entry owns the user-visible state machine and request lock. The
engine entry owns GPU block references. Both are process-local in v1.

## State Machine

Use two-phase commit because the scheduler currently frees request blocks
before `chat_completion_stream_generator()` has finished yielding the HTTP
stream.

```text
READY
  -> GENERATING
  -> ENGINE_COMMITTED
  -> FRONTEND_COMMITTING
  -> READY

Any generation, stream, publish, or discard failure rolls back to the previous
READY head when that head still exists. FAILED is reserved for cases where the
previous head cannot be recovered.
```

1. `READY`
   - A committed engine state may exist.
   - Blocks are pinned by one conversation-owned refcount.
2. `GENERATING`
   - The frontend lock is held.
   - The old `READY` engine state remains pinned.
   - The request has its own refs for reused full blocks and copy-on-write
     blocks.
3. `ENGINE_COMMITTED`
   - The scheduler has captured a pending new engine state before freeing the
     request.
   - The old state is still pinned for rollback.
   - The new state is pinned by transferred request refs.
4. `FRONTEND_COMMITTING`
   - The OpenAI stream has successfully emitted the final chunks.
   - The frontend appends the assistant response to its ledger and asks engine
     core to publish the pending state.
5. `READY`
   - Publish succeeds.
   - The pending state becomes the committed state.
   - Old state refs are released.
6. `FAILED`
   - Only used if the old state is missing or engine publish/discard fails.
   - Normal generation failures roll back to the previous `READY` state.

Failure policy:

- Client disconnect before engine completion: `AsyncLLM.generate()` already
  handles disconnects at `vllm/v1/engine/async_llm.py:583` and calls
  `abort()` at `async_llm.py:588`; discard request state and keep old `READY`.
- Engine finishes but stream fails before `[DONE]`: call
  `discard_conversation_cache_pending(generation_id)` and keep old `READY`.
- Frontend crash after engine pending commit: pending states need TTL cleanup in
  engine core; discard pending and keep old state. If the frontend process
  restarts while the engine survives, v1 accepts that any orphaned committed
  engine states are unreachable from the lost frontend ledger and remain pinned
  only until TTL cleanup.
- A new request for the same id while state is not `READY` returns `409`.

## Exact Cache Mechanics

### Attach

Thread `conversation_id`, `conversation_cache_key`, and `generation_id`
through:

- `ChatCompletionRequest`;
- `EngineClient.generate()`;
- `AsyncLLM.add_request()` and `InputProcessor`;
- `EngineCoreRequest`;
- `Request.from_engine_core_request()`;
- `Request`.

When `Scheduler.add_request()` receives a request with a ready conversation
state:

1. After `Request` construction, but before the request enters the waiting queue
   or any scheduling path can read it, set
   `request.num_computed_tokens = state.token_count`.
2. Populate `req_to_blocks[request_id]` in each cache manager directly from
   the conversation state instead of calling
   `KVCacheManager.get_computed_blocks()`.
3. Mark the request as conversation-attached so `allocate_slots()` does not try
   to discover or cache the attached prefix through normal block hashes.
4. Set `num_cached_block[request_id]` to the number of already attached blocks
   for full-attention groups so `cache_blocks()` does not re-cache them.
5. For Mamba groups in `align` mode, attach the single exact terminal state into
   the physical block-table position that the existing kernel metadata expects:
   `(state.token_count - 1) // block_size`, padded with
   `block_pool.null_block` before it. This matches
   `mamba_get_block_table_tensor()` at
   `vllm/v1/attention/backends/utils.py:887`. This is only placement in vLLM's
   block-addressed tensor; it is not Mamba block caching or prefix matching.

Do not call `MambaManager.find_longest_cache_hit()` for this path.

### Copy-On-Write

Exact append reuse cannot mutate the old committed state before the frontend
commit succeeds.

Therefore attach must use copy-on-write:

- Full-attention groups:
  - Share all fully committed prefix blocks by touching them with
    `BlockPool.touch()`.
  - If `state.token_count % block_size != 0`, allocate one new request-owned
    tail block, copy the committed tail block into it, and use the copy in the
    request block table.
  - If the prefix is block-aligned, no full-attention tail copy is needed.
- Mamba groups:
  - Always attach a request-owned copy of the committed terminal Mamba state
    storage block. Mamba align mode can otherwise write the next turn's running
    state back into the committed terminal state location when the next token
    remains in the same physical block index.

Add a scheduler-to-worker copy list to `SchedulerOutput`:

```python
@dataclass
class ConversationBlockCopy:
    request_id: str
    kv_cache_group_id: int
    src_block_id: int
    dst_block_id: int
    kind: Literal["full_attention_tail", "mamba_terminal"]
```

Run these copies in `GPUModelRunner._update_states()` after
`new_block_ids_to_zero` has been zeroed and before model execution uses the new
block table. For full-attention KV copies, first verify whether
`vllm/_custom_ops.py:2791` supports same-device, same-tensor block-to-block
copies (`swap_blocks(kv, kv, ...)`). If it does not, add a small CUDA
`copy_blocks_in_place` primitive for the KV pool. Treat this as an implementation
step, not a residual risk. For Mamba state copies, reuse
`mamba_utils.collect_mamba_copy_meta()` and `do_mamba_copy_block()` from
`vllm/v1/worker/mamba_utils.py`.

### Generate

Once attached, scheduling and decoding proceed normally:

- prompt tokens before `state.token_count` are treated as already computed;
- suffix prompt tokens and assistant tokens allocate slots as usual;
- `Request.append_output_token_ids()` keeps `_all_token_ids` current at
  `vllm/v1/request.py:200`.

### Engine Commit

Add `Request.conversation_cache_key` and `Request.conversation_generation_id`.

In `Scheduler.update_from_output()`, when a conversation request is finished
successfully:

1. Capture finish reason before free, as existing code already does at
   `scheduler.py:1436`.
2. Before `_free_request(request)` at `scheduler.py:1441`, call a new
   `KVCacheManager.prepare_conversation_commit(request)`.
3. Copy the final request fields needed after metadata teardown into the pending
   state: `request._all_token_ids`, `request.prompt_token_ids`,
   `request.output_token_ids`, `request.num_tokens`, finish reason, and stop
   reason.
4. Transfer request-owned block refs into a pending engine conversation state.
   Do not free those refs.
5. Reuse the existing delayed-free extension point:
   `_free_request(request, delay_free_blocks=True)` at `scheduler.py:1826`.
   This adds the request id to `finished_req_ids` so worker-side
   `CachedRequestState` is cleaned up, and it frees encoder-cache bookkeeping,
   while preventing `_free_blocks(request)` from releasing GPU blocks.
6. After block ownership is transferred, remove scheduler request metadata
   without freeing blocks. Do not leave the finished request parked in
   `self.requests`.
7. Keep the old ready state pinned until frontend publish.

For v1, reject conversation caching when a KV connector / P-D transfer path is
active, because that path also uses `delay_free_blocks=True` and later expects to
call `_free_blocks()` when remote KV transfer completes.

### Exact Terminal Mamba Capture

Existing `mamba_utils.postprocess_mamba()` only copies recurrent state when a
step crosses an aligned full-block boundary. A conversation turn can end at any
token. Add an explicit terminal capture path that is independent of
`MambaManager.find_longest_cache_hit()` and normal prefix-cache block hashes:

```python
@dataclass
class ConversationMambaTerminalCapture:
    request_id: str
    kv_cache_group_id: int
    src_running_state_idx: int
    dst_terminal_state_idx: int
    accept_token_bias: int
```

Requirements:

- Scheduler allocates or identifies one pending terminal state storage block for
  each Mamba group. For compatibility with current worker metadata, that state
  is exposed through the block-table index `(request.num_tokens - 1) //
  block_size`.
- `SchedulerOutput` carries terminal capture metadata.
- `GPUModelRunner._update_states_after_model_execute()` or a nearby worker hook
  runs terminal capture before finished requests are removed at
  `gpu_model_runner.py:1071`.
- Capture must happen even when `request.num_tokens` is not block-aligned.
- After capture, pending engine state owns the terminal Mamba state storage
  block.

This is the key new code that lets us avoid generic Mamba block caching while
still getting exact full-turn recurrent-state reuse. The algorithm is simply:
copy committed terminal state into a request-owned running state, generate the
next turn, then copy the final running state into a new pending committed
terminal state.

### Frontend Commit

Streaming path:

- In `chat_completion_stream_generator()`, collect assistant text and token ids
  internally regardless of `return_token_ids`.
- Add a `stream_succeeded` flag that becomes true only after all engine output
  is consumed and `[DONE]` is yielded without an exception.
- Acquire the per-conversation lock in `OpenAIServingChat.create_chat_completion()`
  or an adjacent cache manager before the generator is returned. The lock object
  lives in the conversation-cache manager, not in the generator frame.
- After `[DONE]`, schedule `conversation_cache.finalize_frontend_commit(...)` as
  a `StreamingResponse` background task from
  `vllm/entrypoints/openai/chat_completion/api_router.py:74` (for example via
  Starlette/FastAPI background task support), not as a bare
  `asyncio.create_task`. This ensures publish runs after the response body has
  completed.
- Hand the background task a release-on-publish-done callback. Keep the
  per-conversation lock until that callback runs.
- The background task calls an engine-client control method:
  `publish_conversation_cache_async(key, generation_id)`.
- If the response stream remains open after the engine has created a pending
  state, the OpenAI layer must keep that pending state alive. Add a lightweight
  `touch_conversation_cache_pending_async(key, generation_id)` control method
  and call it on a timer while the same frontend lock is held, or set the
  pending TTL high enough that normal streaming/backpressure cannot hit it. Use
  the touch method in tests with artificially short TTLs.

Non-streaming path:

- `chat_completion_full_generator()` already collects `final_res` at
  `serving.py:1285`.
- Capture `final_res.outputs[0].token_ids`, append the assistant message, and
  publish the pending engine state before returning the response.
- This can add latency only to non-streaming calls, which are not the voice-loop
  target.

Add engine-client control methods alongside `reset_prefix_cache`:

- `EngineClient.publish_conversation_cache(...)`
- `EngineClient.discard_conversation_cache_pending(...)`
- `EngineClient.touch_conversation_cache_pending(...)`
- `EngineClient.delete_conversation_cache(...)`

Wire them through:

- `vllm/engine/protocol.py`;
- `vllm/v1/engine/async_llm.py`;
- `vllm/v1/engine/core_client.py`;
- `vllm/v1/engine/core.py`;
- `vllm/v1/core/sched/scheduler.py`.

For multiprocess engine mode, use the same utility-call pattern as
`reset_prefix_cache_async()` at `vllm/v1/engine/core_client.py:1086`.

## Multimodal Ledger

Normal vLLM block hashes already include multimodal extra keys, but the exact
conversation cache does not rely on those hashes. We still need deterministic
multimodal metadata so a full prompt can be rendered and split without
reprocessing old audio.

At frontend commit, store:

- committed message count;
- committed prompt token ids;
- for each multimodal item:
  - modality;
  - client UUID, if present;
  - vLLM `identifier`;
  - tokenized placeholder span, as `PlaceholderRange.offset` and
    `PlaceholderRange.length`;
  - base `mm_hash`;
  - enough metadata to recreate placeholders without decoding old media.

On next append:

1. Render/tokenize the request using stored prior multimodal placeholder
   metadata for prefix media.
2. Trust the client-provided message history and committed message count.
3. Split the rendered prompt at `state.token_count`.
4. Classify multimodal features using `feature.mm_position`, which is a
   `PlaceholderRange` (`vllm/multimodal/inputs.py:119`,
   `vllm/multimodal/inputs.py:325`):
   - prefix features must satisfy
     `offset + length <= state.token_count`;
   - suffix features must satisfy `offset >= state.token_count`;
   - any feature spanning the boundary is rejected with `400`, because turn
     boundaries must not split a multimodal placeholder.
5. Send only suffix multimodal features to the engine for encoder processing.
   Conversation-cache attached requests must populate
   `SchedulerOutput.scheduled_encoder_inputs` only for suffix feature indices;
   prefix encoder outputs are not recomputed.
6. Once a turn is committed, prefix encoder outputs are no longer needed for
   future turns because their effect is already captured in the committed
   KV/Mamba state. It is valid for `EncoderCacheManager` to free those outputs;
   add an integration test to prove next-turn correctness after the free.
7. Preserve full prompt token ids in `Request.prompt_token_ids`, because vLLM
   uses the full token row for logits processors, stop checks, and output
   bookkeeping.

This avoids GPU prefill and avoids re-encoding old user audio. If old media is
re-sent and the existing vLLM multimodal cache hits, that is allowed but should
not be required for correctness.

## Request Locking And Invalidation

Locking:

- Use an in-process `asyncio.Lock` per `(served_model, cache_salt,
  conversation_id)`.
- Do not queue by default. If busy, return `409 Conflict` with a `Retry-After`
  header. Use a short value such as `0.1` seconds for
  `ENGINE_COMMITTED`/`FRONTEND_COMMITTING`; use a larger value or omit it while
  the original request is still `GENERATING`.
- Keep the lock from request admission through frontend publish/discard.

Invalidation:

- Add an admin route, for example
  `DELETE /v1/conversations/{conversation_id}/cache`.
- Gate it behind the same OpenAI API key mechanism or a new
  `--enable-conversation-cache-admin` flag for local development.
- Deletion calls frontend ledger removal and engine
  `delete_conversation_cache`.
- Deletion is rejected with `409` if the conversation is generating.

Eviction:

- v1 can default to no automatic eviction for the local single-conversation bot.
- Still add explicit caps:
  - `--conversation-cache-max-conversations`: LRU-evict `READY` entries above
    the cap;
  - `--conversation-cache-max-gpu-blocks`: LRU-evict `READY` entries when pinned
    conversation blocks exceed the cap;
  - `--conversation-cache-pending-ttl-seconds`: discard pending engine states
    that were never published by the frontend;
  - `--conversation-cache-ttl-seconds`: expire inactive `READY` entries.
- If capacity pressure occurs, evict only `READY` entries, never
  `GENERATING` or pending entries.
- Run pending TTL cleanup on the engine-core scheduler loop after a configurable
  interval, not in the OpenAI serving process. Default pending TTL should be
  a leak guard, not a response-flush timer; use a conservative default such as
  600 seconds, and let the frontend touch/extend the pending state while a
  stream is still active. Emit a metric whenever a pending state is discarded
  due to TTL.
- Pinned conversation blocks count against the normal KV pool sized by
  `gpu_memory_utilization` / `--num-gpu-blocks`. Warn when pinned conversation
  blocks exceed a configurable percentage of the pool so long audio sessions do
  not silently starve new allocations.

## Launch Requirements

For Nemotron 3 Nano Omni local voice use:

```bash
vllm serve ... \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-backend triton \
  --max-num-seqs 1 \
  --max-num-batched-tokens <local limit>
```

There is no `--mode-backend` flag in this checkout. The relevant local source
flags are `--mamba-backend` from `vllm/engine/arg_utils.py:868` and
`--attention-backend` from `arg_utils.py:858` if attention backend selection is
needed.

In this checkout, `--mamba-cache-mode align` is not effective unless
`--enable-prefix-caching` is also set; otherwise startup logs warn that Mamba
cache mode has been reset to `none`. On the current local 32 GB GPU with the
other speech service occupying about 3.2 GB, align mode needed
`--gpu-memory-utilization 0.75` for `--max-model-len 4096`; `0.72` failed the
startup KV-capacity check.

Add startup validation:

- If `conversation_id` cache is enabled and the model is hybrid, require
  `--mamba-cache-mode=align`.
- Warn if data parallel size or API server count is greater than 1, because
  v1 conversation state is process-local.
- Smoke-test Triton Mamba kernels on the target RTX 5090 / `sm_120` GPU before
  full-server validation. `MambaConfig.backend` defaults to Triton in
  `vllm/config/mamba.py:36`; if Triton rejects the target architecture, fall
  back to `--mamba-backend flashinfer` where supported or run with
  `--enforce-eager` for diagnosis.
- Confirm conversation-cache attach, COW copies, and terminal Mamba capture run
  outside CUDA graph capture. Verify decode graph behavior and latency with the
  server's CUDA graph settings enabled.
- For Nemotron Omni, optionally add `SupportsMambaPrefixCaching` to
  `NemotronH_Nano_VL_V2` for compatibility, but do not depend on that marker
  for conversation-cache correctness.

Hardware budget note for the target RTX 5090-style 32 GB card:

- NVFP4 model weights leave the KV pool as the main pressure point.
- For one local voice conversation, pinned full-attention prefix blocks plus one
  terminal Mamba state should be well within the KV pool for ordinary sessions;
  terminal Mamba state is expected to be MB-scale, not GB-scale.
- Auto-eviction is not required for the target single-conversation bot, but the
  warning/cap metrics above should catch unusually long sessions.

## Implementation Steps

1. Add schema and request plumbing.
   - Add `conversation_id` to `ChatCompletionRequest`.
   - Add internal conversation fields to `EngineCoreRequest` and `Request`.
   - `EngineCoreRequest` is a `msgspec.Struct` with `array_like=True` at
     `vllm/v1/engine/__init__.py:80`; adding fields changes the internal wire
     format between API and engine-core processes. Keep API server and engine
     core on the same patched build and append fields after existing optional
     fields where possible.
   - Update `InputProcessor` and `AsyncLLM.generate()` plumbing.
   - Tests: schema parse and request field propagation.

2. Add frontend ledger and lock manager.
   - Implement `ConversationFrontendEntry`.
   - Integrate with `OpenAIServingChat.create_chat_completion()`.
   - Reject unsupported options and same-id concurrency.
   - Tests: first request creates no hit; second same-id while locked returns
     `409`.

3. Add engine conversation state manager without reuse.
   - Add publish/discard/delete control methods through `EngineClient`,
     `AsyncLLM`, `CoreClient`, `EngineCore`, and scheduler.
   - Add pending-state lifecycle and TTL cleanup.
   - Copy final token/request fields out of `Request` before scheduler metadata
     teardown.
   - Tests: control methods work in inproc and async-mp clients.

4. Implement exact attach for full-attention KV groups.
   - Add methods on `KVCacheManager` and `SingleTypeKVCacheManager` to attach
     session blocks directly to `req_to_blocks`.
   - Implement full-block sharing and tail copy-on-write.
   - Add worker-side full-attention block copy execution, including a
     same-device same-KV-pool copy primitive if `swap_blocks(kv, kv, ...)` is
     not valid.
   - Tests: refcount ownership, tail COW rollback, text-only two-turn equality,
     same-tensor copy correctness, overlapping source/destination block ids, and
     source equals destination as a no-op.

5. Implement Mamba terminal attach/capture.
   - Add exact-turn Mamba session state storage. Do not use vLLM's Mamba
     block-prefix matching path.
   - Place the one terminal state in the block-table position the existing
     kernels expect, with null padding as needed.
   - Add request-owned terminal copy on attach.
   - Add terminal capture on finish before worker request removal.
   - Tests: Mamba block table index equals `(token_count - 1) // block_size`;
     finish at non-block boundary preserves reusable terminal state.

6. Add frontend streaming/non-streaming finalization.
   - Collect assistant text and token ids in streaming.
   - Publish after successful `[DONE]`; discard on exceptions/disconnects.
   - Use `StreamingResponse` background task support for stream-success publish,
     and touch/extend pending engine states while long streams remain open.
   - Non-streaming publishes after response is built.
   - Tests: stream success, mid-stream disconnect, engine-finished/stream-failed
     rollback.

7. Add multimodal ledger support.
   - Store prior media placeholder metadata and UUIDs.
   - Store tokenized placeholder spans, not just placeholder strings, so prefix
     media can be reconstructed without invoking media decoding.
   - Filter prefix multimodal features so only suffix media is sent to engine.
   - Tests: audio first turn plus text follow-up; two audio turns; no old media
     encoder processing on second turn; prefix encoder outputs can be evicted
     after commit without breaking next-turn correctness.

8. Add admin invalidation and metrics.
   - Delete endpoint.
   - Metrics/logging: hits, misses, busy conflicts, copied tail blocks, copied
     Mamba states, pinned blocks, pending age, TTL discards, evictions.
   - Tests: delete, delete while busy, TTL cleanup, and a long/backpressured
     stream with an artificially short pending TTL must keep its pending state
     alive via touch/extension until publish or discard.

9. Nemotron-specific validation.
   - Run text smoke test through OpenAI-compatible endpoint.
   - Run audio-input smoke test.
   - Compare cached vs uncached output under deterministic sampling.
   - Measure TTFT before/after for multi-turn audio.

## Testing Matrix

Unit tests:

- Conversation key includes model and salt.
- Same id concurrency returns `409`.
- State transitions roll back on failure.
- Full-attention tail block is copied when prefix length is not block-aligned.
- Old ready-state block refs remain pinned until frontend publish.
- Pending refs are released on discard.
- Mamba terminal block index matches token count.
- `delete_conversation_cache` releases refs and removes ledger entries.
- Same-pool KV block copy handles normal copies, overlapping mappings, and
  no-op same-source/destination mappings.

Integration tests with small models/mocks:

- Text-only two-turn deterministic equality versus uncached full prompt.
- Non-block-aligned first turn then append, verifying no mutation of old head.
- Client disconnect before final output leaves old head usable.
- Engine pending state expires if frontend never publishes.
- Pending TTL is extended during a long/backpressured response after the engine
  has created a pending state, then publishes or discards correctly.

Nemotron Omni tests:

- First turn audio input, second turn text input.
- Multiple 5-second audio turns.
- Confirm prompt prefill work is skipped for cached prefix.
- Confirm TTFT improves on second and later turns.
- Confirm generated assistant response is automatically appended to the cache.

Current live harness:

- `scripts/test_vllm_conversation_prefix_cache.py` starts vLLM as a managed
  subprocess by default, waits for `/v1/models`, runs the cache integration
  tests, writes `conversation-prefix-cache-results.json`, and then terminates
  vLLM. Use `--reuse-server` only when intentionally testing an already-running
  endpoint.
- Last managed run command:
  `.venv-vllm-0.20.0-cu132/bin/python scripts/test_vllm_conversation_prefix_cache.py --perf-repeats 2 --long-turns 8`
- Last managed run result: all tests passed. Covered cached vs uncached
  deterministic equivalence, Mamba context-dependent recall, audio-input
  prefix-cache attach through the Omni audio path, TTFT/TPS comparison, and an
  8-turn memory/refcount smoke. Median TTFT was 0.090 s cached vs 0.131 s
  uncached on the long-prefix text case. GPU memory delta during the 8-turn
  smoke was 0 MB.

## Residual Risks

- The suffix rendering path is the most model/template-specific part. Scope v1
  to Nemotron Omni's chat template and local Pipecat client behavior.
- GPU same-device KV block copy may need a small new helper around existing
  lower-level primitives. This must be decided and tested before COW attach is
  implemented.
- Mamba terminal capture is new code and must be tested at non-block-aligned
  turn boundaries.
- CUDA graph behavior must be verified after adding worker-side copy/capture
  hooks. The intended design runs those hooks outside captured decode graphs.
- Process-local cache means multi-replica serving needs sticky routing or a
  shared cache design later.
- Holding old and pending states during two-phase commit temporarily increases
  GPU block pressure. With `--max-num-seqs 1` and local voice use this is
  acceptable; add metrics so we can see it.
- Cold-conversation CPU offload is the natural future path for scaling beyond a
  few local conversations, but it is out of scope for v1.

## Rejected Simplification

A simpler v0 could skip copy-on-write and declare that any stream failure
invalidates the conversation. That would reduce block-copy work, but it would
also make a transient client/network failure corrupt or discard the current
conversation head. The current plan keeps COW because the local voice bot should
survive stream failures without asking the user to restart the conversation.
