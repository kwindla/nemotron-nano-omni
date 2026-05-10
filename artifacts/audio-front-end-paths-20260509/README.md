# Audio Front-End Path Note

This note memorializes the resampling-path findings that matter for the Nemotron audio instability investigation and points to the reproducible tooling added to the repo.

## Main Finding

The live Pipecat input path and the `vLLM` input path do **not** use the same audio dataflow, even though both rely on PyAV / FFmpeg `libswresample`.

- **Pipecat SmallWebRTC path**:
  - receives incoming `AudioFrame`s from the WebRTC track
  - resamples **frame-by-frame** with `av.AudioResampler("s16", "mono", 16000)`
  - converts each processed frame to `np.int16` bytes immediately
  - writes those `16 kHz` PCM16 bytes into the LLM-context WAV
- **`vLLM` path**:
  - decodes the whole input file to a float waveform
  - resamples the **whole buffer** with `av.AudioResampler(format="fltp", layout="mono", rate=16000)`
  - keeps the result as float32
  - feeds that float32 waveform onward without an intermediate PCM16 quantization step

So the relevant comparison is not "Pipecat vs `vLLM` use different libraries." They both use PyAV / `libswresample`. The real differences are:

- sample format: `s16` vs `fltp`
- quantization timing: immediate PCM16 vs no PCM16 at this stage
- granularity: frame-by-frame vs whole-buffer
- end-of-stream handling: SmallWebRTC does not flush a finite resampler at utterance end; `vLLM` does flush and trim

## Production Code Paths

### Pipecat SmallWebRTC

The live bot uses `SmallWebRTCTransport`, not Daily.

- transport selection:
  - [src/nemotron_voice/bot.py](/home/khkramer/src/nemotron-nano-omni/src/nemotron_voice/bot.py:104)
  - [pipecat-core-code/src/pipecat/runner/utils.py](/home/khkramer/src/nemotron-nano-omni/pipecat-core-code/src/pipecat/runner/utils.py:560)
- resampler construction:
  - [pipecat-core-code/src/pipecat/transports/smallwebrtc/transport.py](/home/khkramer/src/nemotron-nano-omni/pipecat-core-code/src/pipecat/transports/smallwebrtc/transport.py:455)
- per-frame resample:
  - [smallwebrtc/transport.py](/home/khkramer/src/nemotron-nano-omni/pipecat-core-code/src/pipecat/transports/smallwebrtc/transport.py:390)
- immediate int16 conversion:
  - [smallwebrtc/transport.py](/home/khkramer/src/nemotron-nano-omni/pipecat-core-code/src/pipecat/transports/smallwebrtc/transport.py:398)
- no further resampling after transport:
  - [src/nemotron_voice/bot.py](/home/khkramer/src/nemotron-nano-omni/src/nemotron_voice/bot.py:325)
  - [pipecat-core-code/src/pipecat/processors/aggregators/llm_context.py](/home/khkramer/src/nemotron-nano-omni/pipecat-core-code/src/pipecat/processors/aggregators/llm_context.py:193)

### `vLLM`

- parser default:
  - [vllm-v0.20.0/vllm/multimodal/parse.py](/home/khkramer/src/nemotron-nano-omni/vllm-v0.20.0/vllm/multimodal/parse.py:563)
- resampler implementation:
  - [vllm-v0.20.0/vllm/multimodal/audio.py](/home/khkramer/src/nemotron-nano-omni/vllm-v0.20.0/vllm/multimodal/audio.py:169)
  - [audio.py](/home/khkramer/src/nemotron-nano-omni/vllm-v0.20.0/vllm/multimodal/audio.py:215)

## Why This Matters

Earlier work showed two important facts:

1. The original real-path Pipecat/browser trace could reproduce a behavior difference relative to the direct `48 kHz -> vLLM` path.
2. A stronger controlled repro showed that even **PCM16 serialization differences after the same PyAV resample** can flip model behavior.

So the working hypothesis is not merely "different front-end libraries." It is:

> small low-level differences in the `16 kHz` waveform that finally reaches the model, including quantization and resampling details, can materially change model behavior.

## New Reproducible Tooling

### 1. Generate path-specific WAVs from a `48 kHz` source file

- [scripts/audio_frontend_path_tools.py](/home/khkramer/src/nemotron-nano-omni/scripts/audio_frontend_path_tools.py:1)
- [scripts/generate_audio_frontend_path_wavs.py](/home/khkramer/src/nemotron-nano-omni/scripts/generate_audio_frontend_path_wavs.py:1)

This generates:

- `*.pipecat-smallwebrtc.wav`
  - `16 kHz`
  - mono
  - PCM16
  - produced by imitating SmallWebRTC's frame-by-frame `AudioResampler("s16", "mono", 16000)` path
- `*.vllm-pyav-float.wav`
  - `16 kHz`
  - mono
  - float WAV
  - produced by imitating `vLLM`'s whole-buffer `AudioResampler(format="fltp", layout="mono", rate=16000)` path

### 2. Sweep promising units across those two paths

- [scripts/run_audio_frontend_path_sweep.py](/home/khkramer/src/nemotron-nano-omni/scripts/run_audio_frontend_path_sweep.py:1)
- promising units manifest:
  - [promising_units.json](/home/khkramer/src/nemotron-nano-omni/artifacts/audio-front-end-paths-20260509/promising_units.json:1)

The sweep:

- reuses the strongest units we care about right now
- generates the Pipecat-style and `vLLM`-style WAVs from the same `48 kHz` source
- compares deterministic real multi-turn inference outputs for:
  - thinking disabled
  - thinking enabled

## Initial Focused Sweep Result

Artifacts:

- sweep report:
  - [artifacts/audio-front-end-path-sweep-20260509/results.json](/home/khkramer/src/nemotron-nano-omni/artifacts/audio-front-end-path-sweep-20260509/results.json:1)
  - [artifacts/audio-front-end-path-sweep-20260509/README.md](/home/khkramer/src/nemotron-nano-omni/artifacts/audio-front-end-path-sweep-20260509/README.md:1)

The initial sweep covered four high-value units:

- `069-backpack__tool_pwd`
- `022-aardvark__text_vowel_count_plus_constant`
- `078-doorknob__text_vowel_count_plus_constant`
- `096-mountain__tool_top_level_file_count`

Deterministic multi-turn result:

- **No second-turn majority differences** appeared between:
  - `source48_original`
  - `pipecat_smallwebrtc`
  - `vllm_pyav_float`
- That held for both:
  - thinking disabled
  - thinking enabled

So for these four units, simply materializing the same source utterance through the two production front-end paths was **not enough** to create a downstream completion-success split.

However, the sweep did still show a meaningful first-turn difference:

- `022-aardvark__text_vowel_count_plus_constant`
- profile: `real_multiturn_thinking`
- `source48_original` and `vllm_pyav_float` first turn:
  - semantically correct aardvark description
- `pipecat_smallwebrtc` first turn:
  - `The auroch was an extinct wild ancestor of domestic cattle...`

So the Pipecat-style path can still move the model off the intended topic, even when the second-turn constrained task happens to land on the same majority answer in this small focused sweep.

The current state is therefore:

- we now have a precise and reproducible implementation of both front-end paths
- we have **not yet** found a strong high-impact conversation-success split that is caused by these two path implementations alone on the four best units we tested
- the most defensible stronger isolated finding in the repo still remains the earlier PCM16 quantization / serialization sensitivity from the original `pwd` repro family

## Example Commands

Generate both path WAVs from a single source file:

```bash
uv run --with av --with soundfile python scripts/generate_audio_frontend_path_wavs.py \
  artifacts/audio-waveform-variant-study-20260504/samples/096-mountain__tool_top_level_file_count/source48_original.wav \
  --out-dir artifacts/audio-front-end-paths-20260509/smoke \
  --basename mountain
```

Run the focused deterministic multi-turn sweep:

```bash
uv run --with numpy --with requests --with av --with soundfile --with scipy --with soxr \
  python scripts/run_audio_frontend_path_sweep.py \
  --overwrite-artifacts \
  --trials 5
```

## Scope / Caveat

These tools isolate the **audio front-end path**. They do not attempt to reproduce:

- live browser/WebRTC transport behavior beyond the resampler and immediate PCM16 conversion
- VAD buffering
- live request timing
- conversation-cache effects

That is intentional. The goal here is the narrowest reproducible comparison:

> given the same `48 kHz` source utterance, what happens if we materialize the `16 kHz` audio exactly the way Pipecat SmallWebRTC does versus the way `vLLM` internally does?
