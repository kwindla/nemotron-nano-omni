# Audio Front-End Bridge Repro

This bundle packages a confirmed bridge repro candidate where:

- the original `48 kHz` source succeeds
- the `vLLM` whole-buffer PyAV path succeeds
- the Pipecat SmallWebRTC frame-by-frame PyAV `s16` path fails

## Case

- Sample id: `022-p01-scared-speed_0p90`
- Transcript: `Tell me in one sentence about a mountain.`
- Emotion: `scared`
- Speed: `0.9`
- Case id: `096-mountain`
- Family id: `tool_top_level_file_count`

## Confirmation

Repeated deterministic replay (`5x` per WAV) produced:

- `source48_original.wav`: `5/5` required tool calls
- `source48_original.vllm-pyav-float.wav`: `5/5` required tool calls
- `source48_original.pipecat-smallwebrtc.wav`: `5/5` blank text, `0/5` tool calls

The observed first-turn texts were then replayed without audio in context. All three text-only analogs produced the required tool call on all `5/5` trials.

That means the failure is not explained by the first-turn wording difference alone; it depends on the audio remaining in context.

## Files

- `source48_original.wav`
- `source48_original.pipecat-smallwebrtc.wav`
- `source48_original.vllm-pyav-float.wav`
- `source48_original.frontend-paths.json`
- `sample_report.json`
- `replay_results.json`
- `text_only_analog_results.json`
- `bundle_summary.json`
