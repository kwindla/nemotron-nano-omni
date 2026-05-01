# Local Nemotron Voice Stack Plan

Goal: evolve the current Nemotron Omni audio bot in small, testable stages. After each stage, the bot should run locally and have an automated smoke test.

## Stage 1: Local Nemotron Speech ASR

Status: completed.

Plan:
- Use `/home/khkramer/src/nemotron-january-2026` as the local ASR server scaffold.
- Update or validate it against the latest NeMo ASR references in `NeMo/nemo/collections/asr/inference` and `NeMo/nemo/agents/voice_agent/pipecat/services/nemo`.
- Run `nvidia/nemotron-speech-streaming-en-0.6b` behind a small WebSocket endpoint.
- Replace the current Pipecat Riva/NIM gRPC STT branch with a WebSocket STT service that consumes this endpoint.
- Add an automated smoke test that sends known audio to the bot and verifies user transcription frames plus the existing Omni response path.

Progress:
- Added `NemotronSpeechWebSocketSTTService`, a Pipecat WebSocket STT client for the local `nemotron_speech.server` protocol.
- Updated the active SmallWebRTC bot to use `NEMOTRON_SPEECH_STT_URL` instead of the Riva/NIM gRPC `NvidiaSTTService`.
- Added `scripts/smoke_step1_asr_bot.py`, which can run the bot against either a real local ASR server or an ASR protocol stub.
- Stub ASR smoke test passed with a known transcript, streamed Omni text, and Cartesia audio output.
- Downloaded `nvidia/nemotron-speech-streaming-en-0.6b` as a local `.nemo` artifact with a 12.5 MB/s cap. Final size: 2,473,041,920 bytes.
- Started the ASR server from `/home/khkramer/src/nemotron-january-2026/src/nemotron_speech/server.py` against the local `.nemo` file.
- Restarted vLLM with `--gpu-memory-utilization 0.78` so the 5090 has enough headroom for the local ASR model.
- Real ASR smoke test passed with `scripts/smoke_step1_asr_bot.py --asr-mode real`; logs show interim/final transcription frames, streamed Omni text, and TTS audio frames.

Exit criteria:
- ASR server health check passes.
- Bot starts with the ASR branch connected.
- aiortc smoke test sends audio and bot logs include a final `TranscriptionFrame`.
- Existing audio-to-Omni-to-TTS loop still produces bot output.

## Stage 2: Pipecat Smart Turn v3

Status: completed.

Plan:
- Find the current Smart Turn v3 examples in `pipecat-core-code/examples`.
- Add Smart Turn v3 to the bot pipeline as the EOU signal.
- Use Smart Turn finalization to trigger ASR final transcript emission and the Nemotron Omni inference turn.
- Keep VAD as the low-level speech activity detector and fallback.

Design notes:
- Pipecat 1.1.0 already defaults `UserTurnStrategies` to `LocalSmartTurnAnalyzerV3`, backed by the bundled `smart-turn-v3.2-cpu.onnx`.
- The current Stage 1 topology puts STT in a branch after the user aggregator, so Smart Turn cannot see `TranscriptionFrame`s.
- Stage 2 should move STT before the user aggregator, then split with a small transcription-only branch so RTVI/client transcription frames still bypass the voice-agent path.

Progress:
- Moved the local Nemotron Speech STT service before the parallel pipeline with `audio_passthrough=True`.
- Kept the primary voice path as audio-input Omni: user aggregator -> audio context collector -> Omni audio LLM -> TTS -> transport output.
- Added a transcription-only parallel branch using `FrameFilter((InterimTranscriptionFrame, TranscriptionFrame))`, so user transcript frames flow directly to RTVI/client observers without going through the voice-agent path.
- Explicitly configured `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())` on the user aggregator.
- Changed the audio context collector to finish on `UserStoppedSpeakingFrame`, so each Omni turn is triggered by Smart Turn finalization after the final transcript has reached the Pipecat context.
- Extended `scripts/smoke_step1_asr_bot.py` with `--require-smart-turn`, which requires log evidence for Smart Turn finalization, `UserStoppedSpeakingFrame`, user audio context insertion, an Omni request with audio parts, LLM text, and TTS audio.
- Real ASR + Smart Turn smoke test passed with `--require-smart-turn`.

Exit criteria:
- Bot starts with Smart Turn v3 enabled.
- Automated smoke test verifies turn finalization without relying only on VAD stop timing.
- Transcription and Omni response still work.

## Stage 3: Local TTS

Status: completed.

Plan:
- Use Kyutai Pocket TTS as the default local TTS path for this single-5090 stack.
- Keep NVIDIA/NeMo Magpie TTS available as a GPU-quality fallback, but do not make it the default because it competes with vLLM for VRAM.
- Run Pocket TTS as a CPU sidecar HTTP service on `127.0.0.1:8001`.
- Replace `CartesiaTTSService` with a local TTS selector that defaults to Pocket TTS and can still select Magpie via `NEMOTRON_TTS_PROVIDER=magpie`.

Progress:
- Cloned `kyutai-labs/pocket-tts` into `pocket-tts/`. Current upstream checkout is `d529606` and package version is `2.0.0`.
- Added `PocketTTSService`, a Pipecat HTTP streaming TTS client for Pocket's `POST /tts` endpoint. It strips the streaming WAV header and emits Pipecat raw TTS audio frames.
- Updated the active SmallWebRTC bot to use `NEMOTRON_TTS_PROVIDER`, defaulting to `pocket`.
- Updated `scripts/smoke_step1_asr_bot.py` so Stage 3 can verify Pocket TTS logs and bot audio output.
- Installed Pocket TTS into `pocket-tts/.venv` via `uv run --project pocket-tts ...`.
- Direct TTS smoke generated `media/pocket-tts-smoke.wav`, a valid 24 kHz mono WAV.
- Full Stage 3 bot smoke passed with stub ASR, real vLLM, Smart Turn v3, and Pocket TTS:
  `scripts/smoke_step1_asr_bot.py --asr-mode stub --require-smart-turn --require-local-tts --tts-provider pocket --max-tokens 48 --client-run-secs 35 --client-silence-secs 5 --log-timeout-secs 90`.
- Pocket TTS is running detached with:
  `setsid -f pocket-tts/.venv/bin/pocket-tts serve --host 127.0.0.1 --port 8001 --quantize > pocket-tts-server.log 2>&1 < /dev/null`.
- Current Pocket TTS process uses about 737 MB RSS and no GPU memory.

Exit criteria:
- Pocket TTS server health check passes.
- Bot starts without Cartesia credentials.
- Automated smoke test verifies bot text frames, bot speech start, and Pocket TTS stream completion.

## Stage 3 Alternate: Kyutai Pocket TTS

Status: superseded by Stage 3 implementation.

Findings:
- Cloned `kyutai-labs/pocket-tts` into `pocket-tts/` for source inspection. Current upstream checkout is `d529606` and package version is `2.0.0`.
- Pocket TTS is a 100M parameter CPU-oriented TTS with a Python API, CLI, FastAPI server, streaming audio generation, predefined voices, and optional int8 quantization.
- Default English model artifacts are small enough for this project: `languages/english/model.safetensors` dry-run reports 219.0 MB, default `alba` voice embedding reports 6.2 MB, and tokenizer reports 59.3 KB.
- The full `kyutai/pocket-tts` repo is gated; the current HF token is denied. The ungated `kyutai/pocket-tts-without-voice-cloning` repo is accessible and is enough for predefined voice embeddings. Arbitrary voice cloning requires accepting the gated model conditions.
- Runtime memory in the upstream quantization docs is about 450 MB baseline or 234 MB with the default int8 quantization profile, on CPU.
- The HTTP `/tts` endpoint streams `audio/wav`; the Python API exposes `generate_audio_stream(...)` yielding 24 kHz mono float tensors. Pipecat integration can either run Pocket TTS in-process and emit `TTSAudioRawFrame`s directly, or run the FastAPI server as a sidecar and strip the WAV header from the chunked response.

Outcome:
- Implemented the sidecar path in Stage 3 with `PocketTTSService`.
- Magpie remains available via `NEMOTRON_TTS_PROVIDER=magpie`.

## Running Notes

- Top-level workspace: `/home/khkramer/src/nemotron-nano-omni`
- Current Omni endpoint: `http://127.0.0.1:8000/v1`
- Current ASR endpoint: `ws://127.0.0.1:8080`
- Current Pocket TTS endpoint: `http://127.0.0.1:8001`
- Current bot file: `pipecat-core-code/examples/function-calling/function-calling-nemotron-omni-audio.py`
- Reference bot snapshot: `pipecat-core-code/examples/function-calling/function-calling-nemotron-omni-audio-context-reference.py`
- Local ASR reference repo: `/home/khkramer/src/nemotron-january-2026`
- Latest NeMo checkout: `/home/khkramer/src/nemotron-nano-omni/NeMo`

## Troubleshooting Notes

- 2026-04-30: A stale bot process was still listening on `7860` and was using the old gRPC `NvidiaSTTService` path, which caused client errors for `127.0.0.1:50051`.
- Restarted the active bot with `NEMOTRON_SPEECH_STT_URL=ws://127.0.0.1:8080`, `NEMOTRON_TTS_PROVIDER=pocket`, and `POCKET_TTS_URL=http://127.0.0.1:8001`.
- Fresh logs show `NemotronSpeechWebSocketSTTService#0` connected to local ASR, interim user transcript frames, and a final `TranscriptionFrame` for `Tell me a story about a unicorn.` reaching the pipeline sink.
- A Playwright load of `http://localhost:7860/client/` after the restart did not reproduce the old `NvidiaSTTService` error; only the expected browser microphone permission warning appeared in that headless session.
