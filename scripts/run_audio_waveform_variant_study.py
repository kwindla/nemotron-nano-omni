#!/usr/bin/env python3
"""Run a fixed-unit audio waveform fragility study over a 50-variant matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import scipy.signal
import soundfile
import soxr

from run_audio_path_fragility_study import (  # noqa: E402
    ALL_TEST_FAMILIES,
    DEFAULT_BASE_URL,
    DEFAULT_ENV_FILES,
    DEFAULT_MODEL,
    DEFAULT_PROBE_DIR,
    PILOT_CASES,
    ROOT,
    SOURCE_AUDIO_SR,
    TARGET_AUDIO_SR,
    Behavior,
    PilotCase,
    TestFamily,
    _audio_file_to_data_url,
    _load_request_template,
    _build_payload,
    _copy_probe_artifacts,
    _ensure_source_audio,
    _load_api_key,
    _normalize_behavior,
    _normalize_plain_text,
    _read_wav_info,
    _resample_audio_pyav,
    _run_payload,
)


DEFAULT_UNIT_MANIFEST = ROOT / "artifacts" / "audio-waveform-variant-study-packaged-units.json"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-waveform-variant-study-20260504"
DEFAULT_VOICE_ID = "71a7ad14-091c-4e8e-a314-022ece01c121"
DEFAULT_TTS_MODEL = "sonic-3"
DEFAULT_CARTESIA_VERSION = "2024-11-13"


@dataclass(frozen=True)
class PackagedUnit:
    case: PilotCase
    family: TestFamily
    bucket: str

    @property
    def unit_id(self) -> str:
        return f"{self.case.case_id}__{self.family.family_id}"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    group: str
    description: str
    builder: Callable[["VariantContext", np.random.Generator], tuple[np.ndarray, int] | Path]


@dataclass
class VariantContext:
    source_path: Path
    source_audio: np.ndarray
    source_sr: int
    pyav16: np.ndarray
    soxr_hq16: np.ndarray
    soxr_mq16: np.ndarray
    scipy_poly16: np.ndarray
    scipy_fft16: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--payload-mode",
        choices=["full_history", "audio_only", "real_multiturn"],
        default="full_history",
        help="Use the original synthetic full-history payload, an audio-only payload, or a real two-turn multi-turn payload.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Override chat_template_kwargs.enable_thinking=true in the payload.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override payload temperature. Default preserves the request template value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override payload top_k. Default preserves the request template value.",
    )
    parser.add_argument(
        "--omit-top-k",
        action="store_true",
        help="Remove top_k from the payload entirely.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override payload top_p. Default preserves the request template value.",
    )
    parser.add_argument(
        "--omit-top-p",
        action="store_true",
        help="Remove top_p from the payload entirely.",
    )
    parser.add_argument("--unit-manifest", type=Path, default=DEFAULT_UNIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--cartesia-version", default=DEFAULT_CARTESIA_VERSION)
    parser.add_argument("--api-key-env", default="CARTESIA_API_KEY")
    parser.add_argument("--env-file", action="append", type=Path)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    return parser.parse_args()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_f32(audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**32)


def _rng_for(case_id: str, variant_id: str) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(case_id, variant_id))


def _quantize_pcm16(
    audio: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    scaled = clipped * 32767.0
    if mode == "trunc":
        values = np.trunc(scaled)
    elif mode == "round":
        values = np.round(scaled)
    elif mode == "dither":
        if rng is None:
            raise ValueError("dither quantization requires an RNG")
        values = np.round(scaled + rng.triangular(-0.5, 0.0, 0.5, size=scaled.shape))
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    return values.astype(np.int16)


def _write_pcm16_wav(path: Path, audio: np.ndarray, *, sample_rate: int, mode: str, rng: np.random.Generator | None = None) -> dict[str, Any]:
    pcm16 = _quantize_pcm16(audio, mode=mode, rng=rng)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    decoded = (pcm16.astype(np.float32) / 32767.0).reshape(-1)
    info = _read_wav_info(path)
    info["sha256_f32"] = _sha256_f32(decoded)
    return info


def _resample_soxr(audio: np.ndarray, *, orig_sr: int, target_sr: int, quality: str) -> np.ndarray:
    resampled = soxr.resample(
        np.asarray(audio, dtype=np.float32),
        in_rate=orig_sr,
        out_rate=target_sr,
        quality=quality,
    )
    return np.asarray(resampled, dtype=np.float32).reshape(-1)


def _resample_scipy_poly(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    result = scipy.signal.resample_poly(np.asarray(audio, dtype=np.float32), up, down)
    return np.asarray(result, dtype=np.float32).reshape(-1)


def _resample_scipy_fft(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    target_len = int(round(len(audio) * target_sr / orig_sr))
    result = scipy.signal.resample(np.asarray(audio, dtype=np.float32), target_len)
    return np.asarray(result, dtype=np.float32).reshape(-1)


def _shift_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples == 0:
        return arr.copy()
    if samples > 0:
        padded = np.concatenate(
            [np.zeros(samples, dtype=np.float32), arr[: max(0, len(arr) - samples)]]
        )
    else:
        shift = abs(samples)
        padded = np.concatenate(
            [arr[shift:], np.zeros(min(shift, len(arr)), dtype=np.float32)]
        )
        if padded.shape[0] < arr.shape[0]:
            padded = np.pad(padded, (0, arr.shape[0] - padded.shape[0]))
    return padded[: arr.shape[0]].astype(np.float32)


def _prepend_silence(audio: np.ndarray, milliseconds: float, sample_rate: int) -> np.ndarray:
    count = int(round(sample_rate * milliseconds / 1000.0))
    return np.concatenate([np.zeros(count, dtype=np.float32), np.asarray(audio, dtype=np.float32)])


def _append_silence(audio: np.ndarray, milliseconds: float, sample_rate: int) -> np.ndarray:
    count = int(round(sample_rate * milliseconds / 1000.0))
    return np.concatenate([np.asarray(audio, dtype=np.float32), np.zeros(count, dtype=np.float32)])


def _apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(np.asarray(audio, dtype=np.float32) * gain, -1.0, 1.0)


def _rms_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr))))
    if rms == 0.0:
        return arr.copy()
    target = 10.0 ** (target_dbfs / 20.0)
    gain = target / rms
    return _apply_gain(arr, gain)


def _add_white_noise(audio: np.ndarray, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    sigma = 10.0 ** (dbfs / 20.0)
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, -1.0, 1.0)


def _add_tpdf_noise(audio: np.ndarray, lsb_scale: float, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    lsb = 1.0 / 32767.0
    noise = rng.triangular(-lsb_scale * lsb, 0.0, lsb_scale * lsb, size=arr.shape)
    return np.clip(arr + noise.astype(np.float32), -1.0, 1.0)


def _add_dc_offset(audio: np.ndarray, offset: float) -> np.ndarray:
    return np.clip(np.asarray(audio, dtype=np.float32) + offset, -1.0, 1.0)


def _build_variant_context(source_path: Path) -> VariantContext:
    source_audio, source_sr = soundfile.read(source_path, dtype="float32", always_2d=False)
    source_sr = int(source_sr)
    arr = np.asarray(source_audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    arr = arr.reshape(-1)
    return VariantContext(
        source_path=source_path,
        source_audio=arr,
        source_sr=source_sr,
        pyav16=_resample_audio_pyav(arr, orig_sr=source_sr, target_sr=TARGET_AUDIO_SR),
        soxr_hq16=_resample_soxr(arr, orig_sr=source_sr, target_sr=TARGET_AUDIO_SR, quality="HQ"),
        soxr_mq16=_resample_soxr(arr, orig_sr=source_sr, target_sr=TARGET_AUDIO_SR, quality="MQ"),
        scipy_poly16=_resample_scipy_poly(arr, orig_sr=source_sr, target_sr=TARGET_AUDIO_SR),
        scipy_fft16=_resample_scipy_fft(arr, orig_sr=source_sr, target_sr=TARGET_AUDIO_SR),
    )


def _baseline_variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            "source48_original",
            "baseline",
            "Original 48 kHz Cartesia WAV passed directly to vLLM.",
            lambda ctx, rng: ctx.source_path,
        ),
        VariantSpec(
            "pyav_trunc16",
            "baseline",
            "PyAV/libswresample 48k->16k, PCM16 truncation.",
            lambda ctx, rng: (ctx.pyav16, TARGET_AUDIO_SR, "trunc"),
        ),
        VariantSpec(
            "pyav_round16",
            "baseline",
            "PyAV/libswresample 48k->16k, PCM16 rounding.",
            lambda ctx, rng: (ctx.pyav16, TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "pyav_dither16",
            "baseline",
            "PyAV/libswresample 48k->16k, PCM16 triangular dither.",
            lambda ctx, rng: (ctx.pyav16, TARGET_AUDIO_SR, "dither"),
        ),
        VariantSpec(
            "soxr_hq_trunc16",
            "baseline",
            "libsoxr HQ 48k->16k, PCM16 truncation.",
            lambda ctx, rng: (ctx.soxr_hq16, TARGET_AUDIO_SR, "trunc"),
        ),
        VariantSpec(
            "soxr_hq_round16",
            "baseline",
            "libsoxr HQ 48k->16k, PCM16 rounding.",
            lambda ctx, rng: (ctx.soxr_hq16, TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "soxr_mq_trunc16",
            "baseline",
            "libsoxr MQ 48k->16k, PCM16 truncation.",
            lambda ctx, rng: (ctx.soxr_mq16, TARGET_AUDIO_SR, "trunc"),
        ),
        VariantSpec(
            "soxr_mq_round16",
            "baseline",
            "libsoxr MQ 48k->16k, PCM16 rounding.",
            lambda ctx, rng: (ctx.soxr_mq16, TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "scipy_polyphase_trunc16",
            "baseline",
            "SciPy polyphase 48k->16k, PCM16 truncation.",
            lambda ctx, rng: (ctx.scipy_poly16, TARGET_AUDIO_SR, "trunc"),
        ),
        VariantSpec(
            "scipy_polyphase_round16",
            "baseline",
            "SciPy polyphase 48k->16k, PCM16 rounding.",
            lambda ctx, rng: (ctx.scipy_poly16, TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "scipy_fft_trunc16",
            "baseline",
            "SciPy FFT 48k->16k, PCM16 truncation.",
            lambda ctx, rng: (ctx.scipy_fft16, TARGET_AUDIO_SR, "trunc"),
        ),
        VariantSpec(
            "scipy_fft_round16",
            "baseline",
            "SciPy FFT 48k->16k, PCM16 rounding.",
            lambda ctx, rng: (ctx.scipy_fft16, TARGET_AUDIO_SR, "round"),
        ),
    ]


def _shift_variant_specs() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for samples in [1, 2, 4, 8, 16]:
        specs.append(
            VariantSpec(
                f"shift_prepend_{samples}samp",
                "sample_shift",
                f"Prepend {samples} zero sample(s), trim the tail to keep length fixed.",
                lambda ctx, rng, n=samples: (_shift_audio(ctx.pyav16, n), TARGET_AUDIO_SR, "round"),
            )
        )
    for samples in [1, 2, 4, 8, 16]:
        specs.append(
            VariantSpec(
                f"shift_drop_{samples}samp",
                "sample_shift",
                f"Drop {samples} leading sample(s), pad zeros at the tail to keep length fixed.",
                lambda ctx, rng, n=samples: (_shift_audio(ctx.pyav16, -n), TARGET_AUDIO_SR, "round"),
            )
        )
    return specs


def _silence_variant_specs() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for ms in [1, 2, 5, 10, 20, 40]:
        specs.append(
            VariantSpec(
                f"prepend_silence_{ms}ms",
                "silence_boundary",
                f"Prepend {ms} ms of silence to the 16 kHz canonical waveform.",
                lambda ctx, rng, m=ms: (_prepend_silence(ctx.pyav16, m, TARGET_AUDIO_SR), TARGET_AUDIO_SR, "round"),
            )
        )
    for ms in [1, 2, 5, 10, 20, 40]:
        specs.append(
            VariantSpec(
                f"append_silence_{ms}ms",
                "silence_boundary",
                f"Append {ms} ms of silence to the 16 kHz canonical waveform.",
                lambda ctx, rng, m=ms: (_append_silence(ctx.pyav16, m, TARGET_AUDIO_SR), TARGET_AUDIO_SR, "round"),
            )
        )
    return specs


def _gain_variant_specs() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for gain in [0.90, 0.94, 0.97, 0.99, 1.01, 1.03, 1.06, 1.10]:
        label = str(gain).replace(".", "p")
        specs.append(
            VariantSpec(
                f"gain_{label}",
                "gain",
                f"Multiply the canonical 16 kHz waveform by gain {gain:.2f}.",
                lambda ctx, rng, g=gain: (_apply_gain(ctx.pyav16, g), TARGET_AUDIO_SR, "round"),
            )
        )
    for dbfs in [-20.0, -16.0]:
        label = f"minus{abs(int(dbfs))}dbfs"
        specs.append(
            VariantSpec(
                f"rms_norm_{label}",
                "gain",
                f"RMS-normalize the canonical 16 kHz waveform to {dbfs:.0f} dBFS.",
                lambda ctx, rng, d=dbfs: (_rms_normalize(ctx.pyav16, d), TARGET_AUDIO_SR, "round"),
            )
        )
    return specs


def _noise_variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            "noise_white_minus60db",
            "noise",
            "Add deterministic white noise at -60 dBFS to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_white_noise(ctx.pyav16, -60.0, rng), TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "noise_white_minus54db",
            "noise",
            "Add deterministic white noise at -54 dBFS to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_white_noise(ctx.pyav16, -54.0, rng), TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "noise_white_minus48db",
            "noise",
            "Add deterministic white noise at -48 dBFS to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_white_noise(ctx.pyav16, -48.0, rng), TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "tpdf_half_lsb",
            "noise",
            "Add deterministic TPDF noise at half an LSB to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_tpdf_noise(ctx.pyav16, 0.5, rng), TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "dc_offset_pos_0p001",
            "noise",
            "Add a +0.001 DC offset to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_dc_offset(ctx.pyav16, 0.001), TARGET_AUDIO_SR, "round"),
        ),
        VariantSpec(
            "dc_offset_neg_0p001",
            "noise",
            "Add a -0.001 DC offset to the canonical 16 kHz waveform.",
            lambda ctx, rng: (_add_dc_offset(ctx.pyav16, -0.001), TARGET_AUDIO_SR, "round"),
        ),
    ]


ALL_VARIANTS: list[VariantSpec] = (
    _baseline_variant_specs()
    + _shift_variant_specs()
    + _silence_variant_specs()
    + _gain_variant_specs()
    + _noise_variant_specs()
)
VARIANT_BY_ID = {variant.variant_id: variant for variant in ALL_VARIANTS}


def _load_packaged_units(manifest_path: Path, *, max_units: int | None) -> list[PackagedUnit]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in PILOT_CASES}
    family_by_id = {family.family_id: family for family in ALL_TEST_FAMILIES}
    units = [
        PackagedUnit(
            case=case_by_id[item["case_id"]],
            family=family_by_id[item["family_id"]],
            bucket=item["bucket"],
        )
        for item in raw
    ]
    if max_units is not None:
        units = units[:max_units]
    return units


def _select_variants(variant_ids: list[str] | None) -> list[VariantSpec]:
    if not variant_ids:
        return ALL_VARIANTS
    return [VARIANT_BY_ID[variant_id] for variant_id in variant_ids]


def _materialize_variant(
    *,
    unit_dir: Path,
    case_id: str,
    variant: VariantSpec,
    ctx: VariantContext,
) -> tuple[Path, dict[str, Any]]:
    path = unit_dir / f"{variant.variant_id}.wav"
    rng = _rng_for(case_id, variant.variant_id)
    built = variant.builder(ctx, rng)
    if isinstance(built, Path):
        shutil.copy2(built, path)
        info = _read_wav_info(path)
        audio, _sr = soundfile.read(path, dtype="float32", always_2d=False)
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        info["sha256_f32"] = _sha256_f32(arr)
        return path, info

    audio, sample_rate, pcm_mode = built
    info = _write_pcm16_wav(path, audio, sample_rate=sample_rate, mode=pcm_mode, rng=rng)
    return path, info


def _summarize_trials(family: TestFamily, case: PilotCase, results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_behavior(family, result) for result in results]
    encoded = [behavior.encoded() for behavior in behaviors]
    prompt_tokens = [result["prompt_tokens"] for result in results]
    counter = Counter(encoded)
    majority_behavior, majority_count = counter.most_common(1)[0]
    expected = family.expected_builder(case).encoded()
    return {
        "trials": len(results),
        "prompt_tokens": prompt_tokens,
        "behaviors": encoded,
        "behavior_counts": dict(sorted(counter.items())),
        "majority_behavior": majority_behavior,
        "majority_count": majority_count,
        "expected_behavior": expected,
        "matches_expected_trials": sum(1 for value in encoded if value == expected),
        "expected_match_rate": sum(1 for value in encoded if value == expected) / len(results),
        "within_variant_stochastic": len(counter) > 1,
    }


def _build_audio_only_payload(case: PilotCase, sample_path: Path, model: str) -> dict[str, Any]:
    payload = _load_request_template()
    payload["model"] = model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["messages"][1]["content"][1]["audio_url"]["url"] = _audio_file_to_data_url(
        sample_path
    )
    payload["messages"] = payload["messages"][:2]
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    return payload


def _build_real_multiturn_payload(
    case: PilotCase,
    family: TestFamily,
    sample_path: Path,
    *,
    model: str,
    first_turn_response: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_request_template()
    payload["model"] = model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["messages"][1]["content"][1]["audio_url"]["url"] = _audio_file_to_data_url(
        sample_path
    )
    payload["messages"] = [
        payload["messages"][0],
        payload["messages"][1],
        {
            "role": "assistant",
            "content": first_turn_response.get("content") or "",
        },
        {
            "role": "user",
            "content": family.prompt_builder(case),
        },
    ]
    return payload


def _apply_template_overrides(
    payload: dict[str, Any],
    *,
    enable_thinking: bool,
) -> None:
    if enable_thinking:
        kwargs = payload.setdefault("chat_template_kwargs", {})
        kwargs["enable_thinking"] = True


def _normalize_audio_only_behavior(result: dict[str, Any]) -> Behavior:
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        return Behavior("tool", tool_calls[0]["function"]["arguments"])
    return Behavior("text", _normalize_plain_text(result.get("content") or ""))


def _summarize_trials_audio_only(results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_audio_only_behavior(result) for result in results]
    encoded = [behavior.encoded() for behavior in behaviors]
    prompt_tokens = [result["prompt_tokens"] for result in results]
    counter = Counter(encoded)
    majority_behavior, majority_count = counter.most_common(1)[0]
    return {
        "trials": len(results),
        "prompt_tokens": prompt_tokens,
        "behaviors": encoded,
        "behavior_counts": dict(sorted(counter.items())),
        "majority_behavior": majority_behavior,
        "majority_count": majority_count,
        "within_variant_stochastic": len(counter) > 1,
    }


def _summarize_first_turn(results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_audio_only_behavior(result) for result in results]
    encoded = [behavior.encoded() for behavior in behaviors]
    counter = Counter(encoded)
    majority_behavior, majority_count = counter.most_common(1)[0]
    return {
        "trials": len(results),
        "behaviors": encoded,
        "behavior_counts": dict(sorted(counter.items())),
        "majority_behavior": majority_behavior,
        "majority_count": majority_count,
        "within_variant_stochastic": len(counter) > 1,
    }


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Audio Waveform Variant Study",
        "",
        "This study runs a fixed set of packaged units across a 50-variant waveform matrix,",
        "with the request shape held constant and the WAV container fixed.",
        "",
        "## Summary",
        "",
        f"- Units: `{report['summary']['units_total']}`",
        f"- Variants: `{report['summary']['variants_total']}`",
        f"- Payload mode: `{report['config']['payload_mode']}`",
        f"- Enable thinking override: `{report['config']['enable_thinking']}`",
        f"- Trials per variant: `{report['config']['trials']}`",
        f"- Temperature override: `{report['config']['temperature']}`",
        f"- top_k override: `{report['config']['top_k']}`",
        f"- omit top_k: `{report['config']['omit_top_k']}`",
        f"- top_p override: `{report['config']['top_p']}`",
        f"- omit top_p: `{report['config']['omit_top_p']}`",
        f"- Total requests: `{report['summary']['total_requests']}`",
        f"- Source48 probe/offline-PyAV float32 matches: "
        f"`{report['summary']['probe_matches_offline_pyav_float32']}/{report['summary']['units_total']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Group | Units differing from `source48_original` majority | Stochastic units |",
        "| --- | --- | ---: | ---: |",
    ]
    for variant_id, summary in report["summary"]["variants"].items():
        lines.append(
            f"| `{variant_id}` | `{summary['group']}` | "
            f"`{summary['majority_differs_from_source48_units']}` | "
            f"`{summary['stochastic_units']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _apply_sampling_overrides(
    payload: dict[str, Any],
    *,
    temperature: float | None,
    top_k: int | None,
    omit_top_k: bool,
    top_p: float | None,
    omit_top_p: bool,
) -> None:
    if temperature is not None:
        payload["temperature"] = temperature
    if omit_top_k:
        payload.pop("top_k", None)
    elif top_k is not None:
        payload["top_k"] = top_k
    if omit_top_p:
        payload.pop("top_p", None)
    elif top_p is not None:
        payload["top_p"] = top_p


def main() -> int:
    args = parse_args()
    if not args.probe_dir.is_dir():
        raise RuntimeError(
            f"Probe directory {args.probe_dir} does not exist. Start vLLM with "
            "VLLM_AUDIO_PROBE_DIR enabled before running this study."
        )
    units = _load_packaged_units(args.unit_manifest, max_units=args.max_units)
    variants = _select_variants(args.variants)

    if args.out_dir.exists() and args.overwrite_artifacts:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_root = args.out_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    env_files = args.env_file or DEFAULT_ENV_FILES
    api_key = _load_api_key(args.api_key_env, env_files)

    manifest = {
        "units": [
            {
                "case_id": unit.case.case_id,
                "topic": unit.case.topic,
                "family_id": unit.family.family_id,
                "bucket": unit.bucket,
            }
            for unit in units
        ],
        "variants": [
            {
                "variant_id": variant.variant_id,
                "group": variant.group,
                "description": variant.description,
            }
            for variant in variants
        ],
    }

    variant_aggregate: dict[str, Counter[str]] = {
        variant.variant_id: Counter() for variant in variants
    }
    unit_reports: list[dict[str, Any]] = []
    probe_matches = 0

    for unit in units:
        unit_dir = samples_root / unit.unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        source_path = _ensure_source_audio(
            case=unit.case,
            case_dir=unit_dir,
            overwrite_audio=args.overwrite_audio,
            api_key=api_key,
            voice_id=args.voice_id,
            tts_model=args.tts_model,
            cartesia_version=args.cartesia_version,
        )
        ctx = _build_variant_context(source_path)
        offline_pyav_hash = _sha256_f32(ctx.pyav16)

        probe_metadata = None
        probe_meta_path = None
        per_variant_results: dict[str, dict[str, Any]] = {}
        source48_majority = None
        prompt_token_drifts = 0

        for variant_idx, variant in enumerate(variants):
            variant_path, variant_info = _materialize_variant(
                unit_dir=unit_dir,
                case_id=unit.case.case_id,
                variant=variant,
                ctx=ctx,
            )
            trial_results: list[dict[str, Any]] = []
            first_turn_results: list[dict[str, Any]] = []
            for trial_idx in range(args.trials):
                need_probe = variant.variant_id == "source48_original" and trial_idx == 0
                if args.payload_mode == "audio_only":
                    payload = _build_audio_only_payload(unit.case, variant_path, args.model)
                    _apply_template_overrides(
                        payload,
                        enable_thinking=args.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        payload,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        omit_top_k=args.omit_top_k,
                        top_p=args.top_p,
                        omit_top_p=args.omit_top_p,
                    )
                    result, metadata, meta_path = _run_payload(
                        base_url=args.base_url,
                        payload=payload,
                        probe_dir=args.probe_dir if need_probe else None,
                    )
                    trial_results.append(result)
                    if need_probe:
                        probe_metadata = metadata
                        probe_meta_path = meta_path
                elif args.payload_mode == "real_multiturn":
                    first_payload = _build_audio_only_payload(unit.case, variant_path, args.model)
                    _apply_template_overrides(
                        first_payload,
                        enable_thinking=args.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        first_payload,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        omit_top_k=args.omit_top_k,
                        top_p=args.top_p,
                        omit_top_p=args.omit_top_p,
                    )
                    first_result, metadata, meta_path = _run_payload(
                        base_url=args.base_url,
                        payload=first_payload,
                        probe_dir=args.probe_dir if need_probe else None,
                    )
                    first_turn_results.append(first_result)
                    if need_probe:
                        probe_metadata = metadata
                        probe_meta_path = meta_path

                    second_payload = _build_real_multiturn_payload(
                        unit.case,
                        unit.family,
                        variant_path,
                        model=args.model,
                        first_turn_response=first_result,
                    )
                    _apply_template_overrides(
                        second_payload,
                        enable_thinking=args.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        second_payload,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        omit_top_k=args.omit_top_k,
                        top_p=args.top_p,
                        omit_top_p=args.omit_top_p,
                    )
                    second_result, _metadata2, _meta_path2 = _run_payload(
                        base_url=args.base_url,
                        payload=second_payload,
                        probe_dir=None,
                    )
                    trial_results.append(second_result)
                else:
                    payload = _build_payload(unit.case, unit.family, variant_path, args.model)
                    _apply_template_overrides(
                        payload,
                        enable_thinking=args.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        payload,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        omit_top_k=args.omit_top_k,
                        top_p=args.top_p,
                        omit_top_p=args.omit_top_p,
                    )
                    result, metadata, meta_path = _run_payload(
                        base_url=args.base_url,
                        payload=payload,
                        probe_dir=args.probe_dir if need_probe else None,
                    )
                    trial_results.append(result)
                    if need_probe:
                        probe_metadata = metadata
                        probe_meta_path = meta_path

            if args.payload_mode == "audio_only":
                summary = _summarize_trials_audio_only(trial_results)
            else:
                summary = _summarize_trials(unit.family, unit.case, trial_results)
            if args.payload_mode == "real_multiturn":
                summary["first_turn"] = _summarize_first_turn(first_turn_results)
            summary["variant_group"] = variant.group
            summary["variant_description"] = variant.description
            summary["file"] = variant_info
            per_variant_results[variant.variant_id] = summary

            if variant.variant_id == "source48_original":
                source48_majority = summary["majority_behavior"]

        if probe_metadata is None or probe_meta_path is None:
            raise RuntimeError(f"Missing source48 probe for {unit.unit_id}")
        copied_probe_meta, copied_probe_wav = _copy_probe_artifacts(probe_meta_path, unit_dir)
        probe_match = probe_metadata["parsed_sha256_f32"] == offline_pyav_hash
        if probe_match:
            probe_matches += 1

        if source48_majority is None:
            raise RuntimeError("source48_original variant is required")

        for variant in variants:
            summary = per_variant_results[variant.variant_id]
            if summary["majority_behavior"] != source48_majority:
                variant_aggregate[variant.variant_id]["majority_differs_from_source48_units"] += 1
            if summary["within_variant_stochastic"]:
                variant_aggregate[variant.variant_id]["stochastic_units"] += 1
            variant_aggregate[variant.variant_id]["units_total"] += 1
            variant_aggregate[variant.variant_id]["trials"] += summary["trials"]
            if "matches_expected_trials" in summary:
                variant_aggregate[variant.variant_id]["matches_expected_trials"] += summary[
                    "matches_expected_trials"
                ]
            if tuple(summary["prompt_tokens"]) != tuple(
                per_variant_results["source48_original"]["prompt_tokens"]
            ):
                prompt_token_drifts += 1

        unit_reports.append(
            {
                "unit": {
                    "unit_id": unit.unit_id,
                    "case_id": unit.case.case_id,
                    "topic": unit.case.topic,
                    "family_id": unit.family.family_id,
                    "bucket": unit.bucket,
                    "audio_prompt": unit.case.audio_prompt,
                    "canonical_reference_response": unit.case.assistant_response,
                },
                "offline_pyav_float32": {
                    "sample_rate": TARGET_AUDIO_SR,
                    "sha256_f32": offline_pyav_hash,
                },
                "probe_source48": {
                    **probe_metadata,
                    "parsed_wav_path": str(copied_probe_wav),
                    "meta_path": str(copied_probe_meta),
                    "matches_offline_pyav_float32": probe_match,
                },
                "source48_majority_behavior": source48_majority,
                "prompt_token_drift_variants": prompt_token_drifts,
                "variants": per_variant_results,
            }
        )

    variant_summary = {
        variant.variant_id: {
            "group": variant.group,
            "description": variant.description,
            "units_total": variant_aggregate[variant.variant_id]["units_total"],
            "majority_differs_from_source48_units": variant_aggregate[variant.variant_id][
                "majority_differs_from_source48_units"
            ],
            "stochastic_units": variant_aggregate[variant.variant_id]["stochastic_units"],
            "trials": variant_aggregate[variant.variant_id]["trials"],
            **(
                {
                    "matches_expected_trials": variant_aggregate[variant.variant_id][
                        "matches_expected_trials"
                    ]
                }
                if args.payload_mode == "full_history"
                else {}
            ),
        }
        for variant in variants
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "payload_mode": args.payload_mode,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "omit_top_k": args.omit_top_k,
            "top_p": args.top_p,
            "omit_top_p": args.omit_top_p,
            "unit_manifest": str(args.unit_manifest),
            "out_dir": str(args.out_dir),
            "probe_dir": str(args.probe_dir),
            "trials": args.trials,
            "voice_id": args.voice_id,
            "tts_model": args.tts_model,
            "cartesia_version": args.cartesia_version,
            "variants": [variant.variant_id for variant in variants],
            "units": [unit.unit_id for unit in units],
        },
        "summary": {
            "units_total": len(units),
            "variants_total": len(variants),
            "total_requests": len(units) * len(variants) * args.trials,
            "probe_matches_offline_pyav_float32": probe_matches,
            "variants": variant_summary,
        },
        "units": unit_reports,
    }

    (args.out_dir / "variant_matrix.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", report)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
