# Audio Front-End Example Set

This bundle packages seven validated examples from the `096-mountain` search:

- two confirmed bridge hits
- five confirmed rescue cases

Each case was checked with:

- repeated deterministic replay of the exact three WAVs
- text-only analog replay using the observed first-turn assistant texts

## Summary

- Examples requested: `7`
- Examples passing sanity check: `7`
- Bridge examples passing: `2`
- Rescue examples passing: `5`

## Examples

### `022-p01-scared-speed_0p90`

- expected pattern: `bridge`
- observed pattern: `{'source48': True, 'pipecat': False, 'vllm': True}`
- sanity check passed: `True`
- transcript: `Tell me in one sentence about a mountain.`
- emotion/profile: `scared` / `speed_0p90`

### `138-p06-excited-speed_1p10`

- expected pattern: `rescue`
- observed pattern: `{'source48': False, 'pipecat': True, 'vllm': False}`
- sanity check passed: `True`
- transcript: `Explain a mountain in one sentence.`
- emotion/profile: `excited` / `speed_1p10`

### `168-p07-sad-speed_1p10`

- expected pattern: `rescue`
- observed pattern: `{'source48': False, 'pipecat': True, 'vllm': False}`
- sanity check passed: `True`
- transcript: `Briefly describe a mountain in one sentence.`
- emotion/profile: `sad` / `speed_1p10`

### `217-p09-sad-speed_0p90`

- expected pattern: `rescue`
- observed pattern: `{'source48': False, 'pipecat': True, 'vllm': False}`
- sanity check passed: `True`
- transcript: `In a single sentence, describe a mountain.`
- emotion/profile: `sad` / `speed_0p90`

### `231-p10-content-baseline`

- expected pattern: `rescue`
- observed pattern: `{'source48': False, 'pipecat': True, 'vllm': False}`
- sanity check passed: `True`
- transcript: `In a single sentence, tell me about a mountain.`
- emotion/profile: `content` / `baseline`

### `478-p20-neutral-speed_1p10`

- expected pattern: `rescue`
- observed pattern: `{'source48': False, 'pipecat': True, 'vllm': False}`
- sanity check passed: `True`
- transcript: `Offer a one-sentence description of a mountain.`
- emotion/profile: `neutral` / `speed_1p10`

### `486-p20-excited-baseline`

- expected pattern: `bridge`
- observed pattern: `{'source48': True, 'pipecat': False, 'vllm': True}`
- sanity check passed: `True`
- transcript: `Offer a one-sentence description of a mountain.`
- emotion/profile: `excited` / `baseline`

## Comparison

- Both bridge cases and all five rescue cases survived repeated deterministic replay: the observed second-turn pattern held for all `5/5` trials per audio variant.
- All seven cases also survived the text-only analog sanity check: when the exact first-turn assistant text was replayed without audio in context, all three paths made the expected tool call on all `5/5` trials.
- That means the failures and rescues are not explained by the observed first-turn wording alone. The audio front-end path remains necessary to produce the second-turn policy split.

## Bridge Cases

- Both bridge cases showed the same second-turn pattern:
  - `source48_original`: correct tool call
  - `pipecat_smallwebrtc`: blank text, no tool call
  - `vllm_pyav_float`: correct tool call
- `486-p20-excited-baseline` is the strongest bridge case in the set because all three paths produced the same first-turn sentence, yet only the Pipecat-style path failed on turn 2.
- `022-p01-scared-speed_0p90` is also a strong bridge case. In that one, the Pipecat-style path produced a different but semantically equivalent first-turn sentence; the text-only analog shows that wording difference does not explain the failure.

## Rescue Cases

- All five rescue cases showed the same second-turn pattern:
  - `source48_original`: blank text, no tool call
  - `pipecat_smallwebrtc`: correct tool call
  - `vllm_pyav_float`: blank text, no tool call
- `138-p06-excited-speed_1p10` is the strongest rescue case because all three paths produced the same first-turn sentence, yet only the Pipecat-style path made the tool call on turn 2.
- `168-p07-sad-speed_1p10`, `217-p09-sad-speed_0p90`, `231-p10-content-baseline`, and `478-p20-neutral-speed_1p10` are also strong rescue cases. In those four, the Pipecat-style path produced a different but semantically equivalent first-turn sentence; the text-only analogs show that wording difference does not explain the rescue.

## Takeaway

- The Pipecat-style front-end path is not simply worse than the `vLLM`-style path.
- In this focused `096-mountain` set, the Pipecat-style path can either harm or improve downstream tool-use success.
- That makes these examples useful for benchmarking because they show a real decision-boundary shift caused by the audio front-end path, not just a one-direction degradation story.
