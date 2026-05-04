# DGX Spark

This platform profile ports the local voice stack to DGX Spark. The shared
application code stays unchanged; the Spark-specific work is in the vLLM patch
stack, BF16 model download flow, and runtime defaults.

The checked-in vLLM patch stack in this repo is Python-only. The standard Spark
environment build therefore uses the repo-local Python checkout together with a
precompiled vLLM extension wheel, instead of recompiling all CUDA extensions
from source on every machine.

## What Is Different

- model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- extra Spark-only vLLM patch:
  `platforms/dgx_spark/patches/vllm-dgx-spark-weight-streaming.patch`
- vendored `C-RADIO` files downloaded directly into the local model directory
- `32k` context length in the standard repo-local startup profile
- `4096` max batched tokens in the standard repo-local startup profile
- single in-flight request by default (`max-num-seqs=1`) for interactive bot
  latency
- exact conversation caching enabled on top of the Spark-safe runtime profile

## Setup

```bash
source platforms/dgx_spark/config/env.sh
platforms/dgx_spark/scripts/apply_vllm_patches.sh
platforms/dgx_spark/scripts/download_model.sh
platforms/dgx_spark/scripts/build_vllm.sh
```

The vLLM source checkout remains `vllm-v0.20.0/`; the Spark patch stack expects
that checkout to be clean before patching. The repo-local vLLM env is
`.venv-vllm-spark/`. If it is stale or partially built, recreate it with:

```bash
NEMOTRON_VLLM_RECREATE_ENV=1 platforms/dgx_spark/scripts/build_vllm.sh
```

`build_vllm.sh` uses a precompiled vLLM wheel payload by default on Spark. To
force a full local source build instead, set:

```bash
export NEMOTRON_VLLM_USE_PRECOMPILED=0
platforms/dgx_spark/scripts/build_vllm.sh
```

## Spark Runtime Notes

DGX Spark uses shared host/device memory. The main bring-up risks are model-load
host-memory spikes and kernel compile spikes. Before starting vLLM:

```bash
free -h
pgrep -af 'vllm|cicc|ptxas|cc1plus|flashinfer'
```

Current defaults in `platforms/dgx_spark/config/env.sh`:

- vLLM native prefix caching disabled; the custom conversation cache must remain
  independent of vLLM's block-prefix cache
- `--mamba-cache-mode align`
- Mamba cache dtype left at vLLM `auto`
- Triton MoE backend
- Triton attention backend
- multimodal profiling disabled
- eager mode enabled
- fixed KV cache budget `3G`
- `32k` max model length
- `4096` max batched tokens
- bot sampling defaults aligned to the Nano Omni model-card instruct mode:
  `temperature=0.2`, `top_k=1`, `max_tokens=1024`, reasoning disabled
- `16` committed conversation-cache entries and `262144` committed
  conversation-cache tokens

The standard startup profile is intentionally conservative because it is the one
we have directly observed booting cleanly from the repo-local vLLM env. Do not
raise `max-model-len`, `max-num-batched-tokens`, or `kv-cache-memory-bytes`
in the checked-in defaults until that larger profile has been revalidated from
this repo.

## Start Services

Standard repo-local vLLM flow:

```bash
source platforms/dgx_spark/config/env.sh
platforms/dgx_spark/scripts/start_vllm.sh check-env
platforms/dgx_spark/scripts/start_vllm.sh start
platforms/dgx_spark/scripts/smoke_vllm.sh
```

`start_vllm.sh start` waits for `/v1/models` to return the configured served
model before it reports success, and it launches the server in a detached
`setsid` session so the pid survives after the wrapper exits. Status codes:

- `0`: vLLM is running and ready
- `1`: vLLM is not running
- `2`: process exists but readiness has not passed yet

Foreground debugging path:

```bash
platforms/dgx_spark/scripts/start_vllm.sh foreground
```

If `start` fails, the script stops the child process, removes the stale pid
file, and prints the tail of `logs/dgx-spark-vllm.log`.

Pocket TTS:

```bash
setsid -f pocket-tts/.venv/bin/pocket-tts serve \
  --host 127.0.0.1 --port 8001 --quantize \
  > logs/dgx-spark-pocket-tts.log 2>&1 < /dev/null
```

ASR:

```bash
setsid -f scripts/start_asr_uv.sh > logs/dgx-spark-asr.log 2>&1 < /dev/null
curl -fsS http://127.0.0.1:8080/health
```

Browser bot:

```bash
NEMOTRON_OMNI_LOG=logs/dgx-spark-bot.log \
NEMOTRON_SPEECH_STT_URL=ws://127.0.0.1:8080 \
NEMOTRON_TTS_PROVIDER=pocket \
POCKET_TTS_URL=http://127.0.0.1:8001 \
PYTHONPATH=$PWD/src \
setsid -f .venv-pipecat/bin/python src/nemotron_voice/bot.py \
  -t webrtc --host 0.0.0.0 --port 7860 \
  > logs/dgx-spark-bot.stdout.log 2>&1 < /dev/null
```

## Validate

Baseline OpenAI smoke:

```bash
platforms/dgx_spark/scripts/smoke_vllm.sh
```

Prefix cache smoke:

```bash
platforms/dgx_spark/scripts/run_prefix_cache_smoke.sh \
  --reuse-server \
  --log "$NEMOTRON_VLLM_LOG" \
  --results-json logs/dgx-spark-prefix-cache-results.json
```

Mixed 20-turn end-to-end regression:

```bash
.venv-pipecat/bin/python scripts/generate_cartesia_audio_fixtures.py --overwrite
.venv-pipecat/bin/python scripts/run_mixed_rtvi_regression.py
```

Direct cached-vs-uncached parity suite against a live vLLM server:

```bash
PYTHONPATH=$PWD/src .venv-pipecat/bin/python scripts/run_direct_cache_state_suite.py \
  --summary-json traces/direct-cache-state-suite-stable.json
```

Alternating live bot benchmark (`10` cached + `10` uncached runs):

```bash
PYTHONPATH=$PWD/src .venv-pipecat/bin/python scripts/run_prefix_cache_benchmark.py --pairs 10
```

This regression keeps one SmallWebRTC session open for 20 turns and mixes:

- RTVI `send-text` turns
- synthetic audio turns sent over the live input audio track
- text-triggered bash-tool turns
- audio-triggered bash-tool turns

It fails if any of the following appear in the fresh bot/vLLM log slice for the
session:

- `409 Conflict`
- `ConversationCacheMissError`
- `attach skipped`
- cache-attach count mismatch between bot `require_cache=True` requests and
  engine `Attached conversation cache` records

The harness uses Cartesia for deterministic spoken fixtures. It looks for
`CARTESIA_API_KEY` in the environment first, then falls back to
`~/src/pipecat/.env` and `~/src/nemotron-speech/.env`.

Pocket TTS parser test:

```bash
pytest tests/test_pocket_tts_wav_streaming.py
```

Shared bot smoke:

```bash
PYTHONPATH=$PWD/src .venv-pipecat/bin/python scripts/smoke_step1_asr_bot.py \
  --asr-mode real \
  --require-smart-turn \
  --require-local-tts \
  --tts-provider pocket \
  --max-tokens 48 \
  --client-run-secs 35 \
  --client-silence-secs 5 \
  --log-timeout-secs 90
```

If the prefix-cache smoke passes but the full bot smoke does not, treat the
problem as service coexistence or runtime budgeting first. The shared Pipecat,
ASR client, and Pocket TTS code paths are intended to remain platform-neutral.

For the control-flow reasoning behind the mixed regression and the cache
contract across Pipecat, the OpenAI frontend, and the engine scheduler, see
`docs/conversation-cache-control-flow.md`.

## Larger Profiles

The repo supports larger Spark-specific overrides, but they are not yet the
documented default startup profile. To test a larger context window, override
the runtime knobs explicitly in your shell before `start_vllm.sh start`, for
example:

```bash
export NEMOTRON_VLLM_MAX_MODEL_LEN=65536
export NEMOTRON_VLLM_MAX_NUM_BATCHED_TOKENS=8192
export NEMOTRON_VLLM_KV_CACHE_MEMORY_BYTES=6G
platforms/dgx_spark/scripts/start_vllm.sh start
```
