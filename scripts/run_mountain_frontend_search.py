#!/usr/bin/env python3
"""Focused 096-mountain search over Cartesia prompt/prosody variants."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_frontend_path_tools import generate_path_wavs
from generate_cartesia_audio_fixtures import (
    DEFAULT_ENV_FILES,
    _load_api_key,
    _synthesize_pcm,
    _write_wav,
)
from run_audio_path_fragility_study import (
    ALL_TEST_FAMILIES,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PILOT_CASES,
    TARGET_AUDIO_SR,
    SOURCE_AUDIO_SR,
    Behavior,
    PilotCase,
    TestFamily,
    _normalize_behavior,
    _run_payload,
)
from run_audio_waveform_variant_study import (
    _apply_sampling_overrides,
    _apply_template_overrides,
    _build_audio_only_payload,
    _build_real_multiturn_payload,
    _normalize_audio_only_behavior,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "mountain-frontend-search-20260509"
DEFAULT_VOICE_ID = "71a7ad14-091c-4e8e-a314-022ece01c121"
DEFAULT_TTS_MODEL = "sonic-3"
DEFAULT_CARTESIA_VERSION = "2024-11-13"

MOUNTAIN_PHRASES = [
    "Tell me in one sentence about a mountain.",
    "Describe a mountain in one sentence.",
    "In one sentence, tell me about a mountain.",
    "Please describe a mountain in one sentence.",
    "Give me one sentence about a mountain.",
    "Explain a mountain in one sentence.",
    "Briefly describe a mountain in one sentence.",
    "Tell me briefly about a mountain in one sentence.",
    "In a single sentence, describe a mountain.",
    "In a single sentence, tell me about a mountain.",
    "Please tell me about a mountain in one sentence.",
    "Share one sentence about a mountain.",
    "Give a one-sentence description of a mountain.",
    "Provide one sentence about a mountain.",
    "Summarize a mountain in one sentence.",
    "What is a mountain? Answer in one sentence.",
    "Describe what a mountain is in one sentence.",
    "Tell me what a mountain is in one sentence.",
    "Please give a one-sentence explanation of a mountain.",
    "Offer a one-sentence description of a mountain.",
]

EMOTIONS = ["neutral", "content", "excited", "sad", "scared"]


@dataclass(frozen=True)
class GenerationProfile:
    profile_id: str
    speed: float | None
    volume: float | None


GENERATION_PROFILES = [
    GenerationProfile("baseline", None, None),
    GenerationProfile("speed_0p90", 0.90, None),
    GenerationProfile("speed_1p10", 1.10, None),
    GenerationProfile("volume_0p85", None, 0.85),
    GenerationProfile("volume_1p15", None, 1.15),
]


@dataclass(frozen=True)
class SearchSample:
    sample_id: str
    transcript: str
    emotion: str
    profile_id: str
    speed: float | None
    volume: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--cartesia-version", default=DEFAULT_CARTESIA_VERSION)
    parser.add_argument("--api-key-env", default="CARTESIA_API_KEY")
    parser.add_argument("--env-file", action="append", type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--cartesia-retries", type=int, default=5)
    return parser.parse_args()


def _build_samples(*, max_samples: int | None) -> list[SearchSample]:
    samples: list[SearchSample] = []
    for phrase_idx, transcript in enumerate(MOUNTAIN_PHRASES, start=1):
        for emotion in EMOTIONS:
            for profile in GENERATION_PROFILES:
                sample_id = (
                    f"{len(samples) + 1:03d}"
                    f"-p{phrase_idx:02d}"
                    f"-{emotion}"
                    f"-{profile.profile_id}"
                )
                samples.append(
                    SearchSample(
                        sample_id=sample_id,
                        transcript=transcript,
                        emotion=emotion,
                        profile_id=profile.profile_id,
                        speed=profile.speed,
                        volume=profile.volume,
                    )
                )
    if max_samples is not None:
        return samples[:max_samples]
    return samples


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Mountain Front-End Search",
        "",
        "Focused search for a `096-mountain` bridge repro using Cartesia source utterance variation",
        "and the two production-like front-end audio paths:",
        "",
        "- `pipecat_smallwebrtc`",
        "- `vllm_pyav_float`",
        "",
        "All inference uses deterministic real multi-turn settings:",
        "",
        "- thinking enabled",
        "- `temperature=0`",
        "- `top_k` omitted",
        "",
        "## Summary",
        "",
        f"- Samples: `{report['summary']['samples_total']}`",
        f"- Source utterances generated: `{report['summary']['samples_total']}`",
        f"- Variants per sample: `3`",
        f"- Total model requests: `{report['summary']['total_requests']}`",
        f"- Bridge hits: `{report['summary']['bridge_hits_total']}`",
        f"- Reverse bridge hits: `{report['summary']['reverse_bridge_hits_total']}`",
        f"- Any second-turn diffs: `{report['summary']['second_turn_diff_total']}`",
        f"- Any first-turn diffs: `{report['summary']['first_turn_diff_total']}`",
        "",
        "## Bridge Hits",
        "",
        "A bridge hit means:",
        "",
        "- `source48_original` second turn matched expected",
        "- `vllm_pyav_float` second turn matched expected",
        "- `pipecat_smallwebrtc` second turn did not match expected",
        "",
    ]
    if report["summary"]["bridge_hit_ids"]:
        for sample_id in report["summary"]["bridge_hit_ids"]:
            lines.append(f"- `{sample_id}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.extend(
        [
            "## Reverse Bridge Hits",
            "",
            "A reverse bridge hit means:",
            "",
            "- `source48_original` second turn did not match expected",
            "- `vllm_pyav_float` second turn did not match expected",
            "- `pipecat_smallwebrtc` second turn matched expected",
            "",
        ]
    )
    if report["summary"]["reverse_bridge_hit_ids"]:
        for sample_id in report["summary"]["reverse_bridge_hit_ids"]:
            lines.append(f"- `{sample_id}`")
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_mountain_case() -> PilotCase:
    return next(case for case in PILOT_CASES if case.case_id == "096-mountain")


def _find_mountain_family() -> TestFamily:
    return next(family for family in ALL_TEST_FAMILIES if family.family_id == "tool_top_level_file_count")


def _ensure_source_audio(
    sample: SearchSample,
    sample_dir: Path,
    *,
    api_key: str,
    voice_id: str,
    model: str,
    cartesia_version: str,
    overwrite_audio: bool,
    cartesia_retries: int,
) -> Path:
    path = sample_dir / "source48_original.wav"
    if path.exists() and not overwrite_audio:
        return path
    last_error: Exception | None = None
    for attempt in range(1, cartesia_retries + 1):
        try:
            pcm = _synthesize_pcm(
                api_key=api_key,
                transcript=sample.transcript,
                voice_id=voice_id,
                model=model,
                sample_rate=SOURCE_AUDIO_SR,
                cartesia_version=cartesia_version,
                speed=sample.speed,
                volume=sample.volume,
                emotion=sample.emotion,
            )
            break
        except (RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= cartesia_retries:
                raise
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    else:
        assert last_error is not None
        raise last_error
    _write_wav(path, pcm, sample_rate=SOURCE_AUDIO_SR)
    return path


def _first_turn_summary(result: dict[str, Any]) -> dict[str, Any]:
    behavior = _normalize_audio_only_behavior(result)
    return {
        "behavior": behavior.encoded(),
        "content": result.get("content") or "",
        "tool_calls": result.get("tool_calls") or [],
        "prompt_tokens": result["prompt_tokens"],
    }


def _second_turn_summary(case: PilotCase, family: TestFamily, result: dict[str, Any]) -> dict[str, Any]:
    behavior = _normalize_behavior(family, result)
    expected = family.expected_builder(case).encoded()
    return {
        "behavior": behavior.encoded(),
        "content": result.get("content") or "",
        "tool_calls": result.get("tool_calls") or [],
        "prompt_tokens": result["prompt_tokens"],
        "expected_behavior": expected,
        "matches_expected": behavior.encoded() == expected,
    }


def _sample_report_path(sample_dir: Path) -> Path:
    return sample_dir / "sample_report.json"


def _reverse_bridge_hit(report: dict[str, Any]) -> bool:
    if "reverse_bridge_hit" in report:
        return bool(report["reverse_bridge_hit"])
    variants = report["variants"]
    return (
        not variants["source48_original"]["second_turn"]["matches_expected"]
        and not variants["vllm_pyav_float"]["second_turn"]["matches_expected"]
        and variants["pipecat_smallwebrtc"]["second_turn"]["matches_expected"]
    )


def _accumulate_counts(
    report: dict[str, Any],
    *,
    bridge_hit_ids: list[str],
    reverse_bridge_hit_ids: list[str],
) -> tuple[int, int]:
    first_turn_diff = 1 if report["first_turn_diff"] else 0
    second_turn_diff = 1 if report["second_turn_diff"] else 0
    if report["bridge_hit"]:
        bridge_hit_ids.append(report["sample"]["sample_id"])
    if _reverse_bridge_hit(report):
        reverse_bridge_hit_ids.append(report["sample"]["sample_id"])
    return first_turn_diff, second_turn_diff


def _write_progress(
    *,
    out_dir: Path,
    samples_total: int,
    completed_reports: list[dict[str, Any]],
    bridge_hit_ids: list[str],
    reverse_bridge_hit_ids: list[str],
    first_turn_diff_total: int,
    second_turn_diff_total: int,
) -> None:
    progress = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "samples_total": samples_total,
        "samples_completed": len(completed_reports),
        "samples_remaining": samples_total - len(completed_reports),
        "bridge_hits_total": len(bridge_hit_ids),
        "bridge_hit_ids": bridge_hit_ids,
        "reverse_bridge_hits_total": len(reverse_bridge_hit_ids),
        "reverse_bridge_hit_ids": reverse_bridge_hit_ids,
        "first_turn_diff_total": first_turn_diff_total,
        "second_turn_diff_total": second_turn_diff_total,
        "completed_sample_ids": [report["sample"]["sample_id"] for report in completed_reports],
    }
    (out_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite_artifacts:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env_files = args.env_file or DEFAULT_ENV_FILES
    api_key = _load_api_key(args.api_key_env, env_files)
    case = _find_mountain_case()
    family = _find_mountain_family()
    samples = _build_samples(max_samples=args.max_samples)
    (args.out_dir / "source_manifest.json").write_text(
        json.dumps([asdict(sample) for sample in samples], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    unit_reports: list[dict[str, Any]] = []
    bridge_hit_ids: list[str] = []
    reverse_bridge_hit_ids: list[str] = []
    second_turn_diff_total = 0
    first_turn_diff_total = 0

    for sample in samples:
        sample_dir = args.out_dir / "samples" / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_report_path = _sample_report_path(sample_dir)
        if sample_report_path.is_file():
            sample_report = json.loads(sample_report_path.read_text(encoding="utf-8"))
            unit_reports.append(sample_report)
            first_inc, second_inc = _accumulate_counts(
                sample_report,
                bridge_hit_ids=bridge_hit_ids,
                reverse_bridge_hit_ids=reverse_bridge_hit_ids,
            )
            first_turn_diff_total += first_inc
            second_turn_diff_total += second_inc
            _write_progress(
                out_dir=args.out_dir,
                samples_total=len(samples),
                completed_reports=unit_reports,
                bridge_hit_ids=bridge_hit_ids,
                reverse_bridge_hit_ids=reverse_bridge_hit_ids,
                first_turn_diff_total=first_turn_diff_total,
                second_turn_diff_total=second_turn_diff_total,
            )
            continue
        source_wav = _ensure_source_audio(
            sample,
            sample_dir,
            api_key=api_key,
            voice_id=args.voice_id,
            model=args.tts_model,
            cartesia_version=args.cartesia_version,
            overwrite_audio=args.overwrite_audio,
            cartesia_retries=args.cartesia_retries,
        )
        generated = generate_path_wavs(source_wav, sample_dir, basename="source48_original")
        variants = {
            "source48_original": source_wav,
            "pipecat_smallwebrtc": generated.pipecat_wav,
            "vllm_pyav_float": generated.vllm_wav,
        }

        variant_reports: dict[str, dict[str, Any]] = {}
        for variant_id, sample_path in variants.items():
            first_payload = _build_audio_only_payload(case, sample_path, args.model)
            _apply_template_overrides(first_payload, enable_thinking=True)
            _apply_sampling_overrides(
                first_payload,
                temperature=0.0,
                top_k=None,
                omit_top_k=True,
                top_p=None,
                omit_top_p=False,
            )
            first_result, _metadata, _meta_path = _run_payload(
                base_url=args.base_url,
                payload=first_payload,
                probe_dir=None,
            )

            second_payload = _build_real_multiturn_payload(
                case,
                family,
                sample_path,
                model=args.model,
                first_turn_response=first_result,
            )
            _apply_template_overrides(second_payload, enable_thinking=True)
            _apply_sampling_overrides(
                second_payload,
                temperature=0.0,
                top_k=None,
                omit_top_k=True,
                top_p=None,
                omit_top_p=False,
            )
            second_result, _metadata2, _meta_path2 = _run_payload(
                base_url=args.base_url,
                payload=second_payload,
                probe_dir=None,
            )
            variant_reports[variant_id] = {
                "sample_path": str(sample_path),
                "first_turn": _first_turn_summary(first_result),
                "second_turn": _second_turn_summary(case, family, second_result),
            }

        first_behaviors = {
            variant_id: variant["first_turn"]["behavior"] for variant_id, variant in variant_reports.items()
        }
        second_behaviors = {
            variant_id: variant["second_turn"]["behavior"] for variant_id, variant in variant_reports.items()
        }
        first_turn_diff = len(set(first_behaviors.values())) > 1
        second_turn_diff = len(set(second_behaviors.values())) > 1
        bridge_hit = (
            variant_reports["source48_original"]["second_turn"]["matches_expected"]
            and variant_reports["vllm_pyav_float"]["second_turn"]["matches_expected"]
            and not variant_reports["pipecat_smallwebrtc"]["second_turn"]["matches_expected"]
        )
        reverse_bridge_hit = (
            not variant_reports["source48_original"]["second_turn"]["matches_expected"]
            and not variant_reports["vllm_pyav_float"]["second_turn"]["matches_expected"]
            and variant_reports["pipecat_smallwebrtc"]["second_turn"]["matches_expected"]
        )

        sample_report = {
            "sample": asdict(sample),
            "variants": variant_reports,
            "first_turn_diff": first_turn_diff,
            "second_turn_diff": second_turn_diff,
            "bridge_hit": bridge_hit,
            "reverse_bridge_hit": reverse_bridge_hit,
        }
        sample_report_path.write_text(
            json.dumps(sample_report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        unit_reports.append(sample_report)
        first_inc, second_inc = _accumulate_counts(
            sample_report,
            bridge_hit_ids=bridge_hit_ids,
            reverse_bridge_hit_ids=reverse_bridge_hit_ids,
        )
        first_turn_diff_total += first_inc
        second_turn_diff_total += second_inc
        _write_progress(
            out_dir=args.out_dir,
            samples_total=len(samples),
            completed_reports=unit_reports,
            bridge_hit_ids=bridge_hit_ids,
            reverse_bridge_hit_ids=reverse_bridge_hit_ids,
            first_turn_diff_total=first_turn_diff_total,
            second_turn_diff_total=second_turn_diff_total,
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "voice_id": args.voice_id,
            "tts_model": args.tts_model,
            "cartesia_version": args.cartesia_version,
            "out_dir": str(args.out_dir),
            "max_samples": args.max_samples,
            "profile": {
                "payload_mode": "real_multiturn",
                "enable_thinking": True,
                "temperature": 0.0,
                "omit_top_k": True,
            },
        },
        "summary": {
            "samples_total": len(samples),
            "total_requests": len(samples) * 3 * 2,
            "bridge_hits_total": len(bridge_hit_ids),
            "bridge_hit_ids": bridge_hit_ids,
            "reverse_bridge_hits_total": len(reverse_bridge_hit_ids),
            "reverse_bridge_hit_ids": reverse_bridge_hit_ids,
            "first_turn_diff_total": first_turn_diff_total,
            "second_turn_diff_total": second_turn_diff_total,
        },
        "samples": unit_reports,
    }

    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
