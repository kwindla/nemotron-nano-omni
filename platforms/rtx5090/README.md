# RTX 5090

This platform profile is the current NVFP4 local voice stack for an RTX 5090.
It keeps the original single-sequence browser-bot settings and the exact
conversation-cache patch path.

## Setup

```bash
source platforms/rtx5090/config/env.sh
platforms/rtx5090/scripts/apply_vllm_patches.sh
python3 platforms/common/scripts/patch_nemotron_chat_template.py "$NEMOTRON_MODEL_PATH"
```

The env file sets the shared defaults used by:

- `src/nemotron_voice/bot.py`
- `scripts/test_vllm_conversation_prefix_cache.py`
- `platforms/rtx5090/scripts/start_vllm.sh`

## Runtime Defaults

- model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- served model name: `nemotron_3_nano_omni`
- context length: `32768`
- sequence budget: `1`
- max batched tokens: `8192`
- vLLM native prefix caching disabled; the custom conversation cache must remain
  independent of vLLM's block-prefix cache
- Mamba cache mode: `align`
- Mamba cache dtype: `float32`; RTX NVFP4 direct parity drifted with the default
  lower-precision Mamba cache state, and `float32` requires
  `max-num-batched-tokens >= 4240`
- KV cache budget: no explicit `NEMOTRON_VLLM_KV_CACHE_MEMORY_BYTES` by default;
  the previous `2G` setting was only a tight-memory diagnostic profile
- bot sampling defaults aligned to the Nano Omni model-card instruct mode:
  `temperature=0.2`, `top_k=1`, `max_tokens=1024`, reasoning disabled
- exact conversation cache enabled with the 5090 headroom settings from the
  original browser runbook
- the model chat template is patched so assistant tool-call history renders
  `</tool_call><|im_end|>` without an extra newline; this keeps generated
  tool-call tokens round-trippable through the next cached prompt render

## Start Services

ASR:

```bash
setsid -f scripts/start_asr_uv.sh > logs/rtx5090-asr.log 2>&1 < /dev/null
curl -fsS http://127.0.0.1:8080/health
```

Pocket TTS:

```bash
setsid -f pocket-tts/.venv/bin/pocket-tts serve \
  --host 127.0.0.1 --port 8001 --quantize \
  > logs/rtx5090-pocket-tts.log 2>&1 < /dev/null
```

vLLM:

```bash
platforms/rtx5090/scripts/start_vllm.sh check-env
platforms/rtx5090/scripts/start_vllm.sh start
```

Use `foreground` instead of `start` when you want logs in the current shell.

Browser bot:

```bash
NEMOTRON_OMNI_LOG=logs/rtx5090-bot.log \
NEMOTRON_SPEECH_STT_URL=ws://127.0.0.1:8080 \
NEMOTRON_TTS_PROVIDER=pocket \
POCKET_TTS_URL=http://127.0.0.1:8001 \
PYTHONPATH=$PWD/src \
setsid -f .venv-pipecat/bin/python src/nemotron_voice/bot.py \
  -t webrtc --host 0.0.0.0 --port 7860 \
  > logs/rtx5090-bot.stdout.log 2>&1 < /dev/null
```

## Validate

Prefix cache smoke:

```bash
platforms/rtx5090/scripts/run_prefix_cache_smoke.sh \
  --reuse-server \
  --log "$NEMOTRON_VLLM_LOG" \
  --results-json logs/rtx5090-prefix-cache-results.json
```

Mixed 20-turn end-to-end benchmark:

```bash
PYTHONPATH=$PWD/src .venv-pipecat/bin/python scripts/run_prefix_cache_benchmark.py \
  --platform rtx5090 \
  --pairs 10
```

Shared bot smoke:

```bash
PYTHONPATH=$PWD/src .venv-pipecat/bin/python scripts/smoke_step1_asr_bot.py \
  --asr-mode stub \
  --require-smart-turn \
  --require-local-tts \
  --tts-provider pocket \
  --max-tokens 48 \
  --client-run-secs 35 \
  --client-silence-secs 5 \
  --log-timeout-secs 90
```
