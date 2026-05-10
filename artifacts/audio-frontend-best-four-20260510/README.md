# Audio Front-End Best Four

This bundle is the smallest high-signal subset from the validated `096-mountain` example set.

It contains:

- two strongest bridge cases
- two strongest rescue cases

Each case was already validated with:

- repeated deterministic replay of the exact three WAVs (`5/5` trials per variant)
- text-only analog replay using the observed first-turn assistant texts (`5/5` successful tool calls for all three paths)

That means the second-turn tool-policy split is not explained by the first-turn wording alone.

## Summary

- Examples requested: `4`
- Examples passing sanity check: `4`
- Bridge examples passing: `2`
- Rescue examples passing: `2`

## Selection Rule

These four were chosen by causal cleanliness:

1. deterministic replay stability
2. clean text-only sanity check
3. minimal first-turn confounds
4. realistic generation profile
5. explanatory clarity

## Bridge Cases

### `486-p20-excited-baseline`

- strongest bridge case in the set
- transcript: `Offer a one-sentence description of a mountain.`
- all three paths produced the same first-turn sentence
- second-turn pattern:
  - `source48_original`: correct tool call
  - `vllm_pyav_float`: correct tool call
  - `pipecat_smallwebrtc`: blank text

### `022-p01-scared-speed_0p90`

- second strongest bridge case
- transcript: `Tell me in one sentence about a mountain.`
- `source48` and `vllm` succeeded
- `pipecat` failed with blank text
- weaker than `486` only because the Pipecat-style path changed the first-turn wording, though the text-only analog confirms that wording change does not explain the failure

## Rescue Cases

### `138-p06-excited-speed_1p10`

- strongest rescue case in the set
- transcript: `Explain a mountain in one sentence.`
- all three paths produced the same first-turn sentence
- second-turn pattern:
  - `source48_original`: blank text
  - `vllm_pyav_float`: blank text
  - `pipecat_smallwebrtc`: correct tool call

### `231-p10-content-baseline`

- strongest realistic rescue case after `138`
- transcript: `In a single sentence, tell me about a mountain.`
- baseline profile
- `source48` and `vllm` failed with blank text
- `pipecat` made the required tool call
- the Pipecat-style path used a semantically equivalent first-turn sentence, and the text-only analog confirms that wording difference does not explain the rescue

## Takeaway

- The Pipecat-style front-end path can deterministically hurt downstream tool-use success.
- The same path can also deterministically rescue downstream tool-use success.
- These four examples are the clearest compact benchmark set we currently have for showing that the audio front-end path changes the model's decision region, not just the surface wording of the first response.
