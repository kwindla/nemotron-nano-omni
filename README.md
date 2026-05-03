# Nemotron Nano Omni Local Voice Stack

This repo now treats hardware bring-up as a platform problem, not an application
fork.

Shared application code stays in:

- `src/` for the Pipecat bot, local Nemotron Speech ASR server, and TTS/LLM services
- `scripts/` for shared smoke harnesses
- `tests/` for small unit tests

Platform-specific runbooks, configs, scripts, and extra patches live in:

- `platforms/rtx5090/` for the current NVFP4 + RTX 5090 stack
- `platforms/dgx_spark/` for the BF16 + DGX Spark stack
- `platforms/common/patches/` for patch files shared by both platforms

## Supported Platforms

- [RTX 5090](/home/khkramer/src/nemotron-nano-omni/platforms/rtx5090/README.md)
- [DGX Spark](/home/khkramer/src/nemotron-nano-omni/platforms/dgx_spark/README.md)

## Shared Behavior

Both platforms use the same voice-agent code path:

- Pipecat handles transport, Smart Turn, context, and tool calling.
- Nemotron Speech provides local streaming ASR over WebSocket.
- Kyutai Pocket TTS remains the default local TTS path.
- The custom vLLM conversation cache patch provides exact append-only prefix
  reuse with `conversation_id`.

The shared bot no longer hardcodes one served model id. Set
`NEMOTRON_OMNI_MODEL` or source one of the platform env files before starting
the bot.

## Platform Workflow

Each platform folder contains a sourceable env file plus wrapper scripts. The
expected workflow is:

```bash
source platforms/<platform>/config/env.sh
platforms/<platform>/scripts/apply_vllm_patches.sh
```

From there:

- use the platform `build_vllm.sh`, `start_vllm.sh`, and smoke wrappers where
  they exist
- use the shared bot and ASR/TTS smokes from `scripts/`

The shared vLLM checkout is still pinned at `v0.20.0` in
[checkouts.lock.json](/home/khkramer/src/nemotron-nano-omni/checkouts.lock.json).
Platform patch stacks are additive on top of that checkout, so switching from
one platform patch stack to another is easiest with a fresh `vllm-v0.20.0`
checkout.

## Shared Validation

Common test entry points:

- `pytest tests/test_pocket_tts_wav_streaming.py`
- `scripts/test_vllm_conversation_prefix_cache.py`
- `scripts/smoke_step1_asr_bot.py`

The platform env files provide the right defaults for model path, served model
name, vLLM binary path, context length, and cache/runtime flags.
