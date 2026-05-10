#!/usr/bin/env python3
"""Generate Pipecat/vLLM path WAVs and compare deterministic real multi-turn outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_frontend_path_tools import generate_path_wavs
from run_audio_path_fragility_study import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    Behavior,
    _normalize_behavior,
    _run_payload,
)
from run_audio_waveform_variant_study import (
    DEFAULT_UNIT_MANIFEST,
    PackagedUnit,
    _apply_sampling_overrides,
    _apply_template_overrides,
    _build_audio_only_payload,
    _build_real_multiturn_payload,
    _load_packaged_units,
    _normalize_audio_only_behavior,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMISING_UNITS = ROOT / "artifacts" / "audio-front-end-paths-20260509" / "promising_units.json"
DEFAULT_SOURCE_STUDY_ROOT = ROOT / "artifacts" / "audio-waveform-variant-study-20260504" / "samples"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-front-end-path-sweep-20260509"


@dataclass(frozen=True)
class SweepProfile:
    profile_id: str
    enable_thinking: bool
    temperature: float
    omit_top_k: bool


PROFILES = [
    SweepProfile(
        profile_id="real_multiturn_no_thinking",
        enable_thinking=False,
        temperature=0.0,
        omit_top_k=True,
    ),
    SweepProfile(
        profile_id="real_multiturn_thinking",
        enable_thinking=True,
        temperature=0.0,
        omit_top_k=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--unit-manifest", type=Path, default=DEFAULT_PROMISING_UNITS)
    parser.add_argument("--packaged-unit-manifest", type=Path, default=DEFAULT_UNIT_MANIFEST)
    parser.add_argument("--source-study-root", type=Path, default=DEFAULT_SOURCE_STUDY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--overwrite-artifacts", action="store_true")
    return parser.parse_args()


def _summarize_behavior_counts(behaviors: list[Behavior]) -> dict[str, Any]:
    encoded = [behavior.encoded() for behavior in behaviors]
    counter = Counter(encoded)
    majority, majority_count = counter.most_common(1)[0]
    return {
        "behaviors": encoded,
        "behavior_counts": dict(sorted(counter.items())),
        "majority_behavior": majority,
        "majority_count": majority_count,
        "within_variant_stochastic": len(counter) > 1,
    }


def _summarize_first_turn(results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_audio_only_behavior(result) for result in results]
    summary = _summarize_behavior_counts(behaviors)
    summary["prompt_tokens"] = [result["prompt_tokens"] for result in results]
    summary["raw_results"] = [
        {
            "content": result.get("content") or "",
            "tool_calls": result.get("tool_calls") or [],
            "prompt_tokens": result["prompt_tokens"],
        }
        for result in results
    ]
    return summary


def _summarize_second_turn(unit: PackagedUnit, results: list[dict[str, Any]]) -> dict[str, Any]:
    behaviors = [_normalize_behavior(unit.family, result) for result in results]
    summary = _summarize_behavior_counts(behaviors)
    expected = unit.family.expected_builder(unit.case).encoded()
    summary["expected_behavior"] = expected
    summary["matches_expected_trials"] = sum(
        1 for behavior in summary["behaviors"] if behavior == expected
    )
    summary["expected_match_rate"] = summary["matches_expected_trials"] / len(results)
    summary["prompt_tokens"] = [result["prompt_tokens"] for result in results]
    summary["raw_results"] = [
        {
            "content": result.get("content") or "",
            "tool_calls": result.get("tool_calls") or [],
            "prompt_tokens": result["prompt_tokens"],
        }
        for result in results
    ]
    return summary


def _find_source_wav(unit: PackagedUnit, source_study_root: Path) -> Path:
    path = source_study_root / unit.unit_id / "source48_original.wav"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source48_original.wav for {unit.unit_id}: {path}")
    return path


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Audio Front-End Path Sweep",
        "",
        "This sweep generates three audio inputs per unit:",
        "- `source48_original`: the original 48 kHz WAV",
        "- `pipecat_smallwebrtc`: a 16 kHz PCM16 WAV generated with Pipecat SmallWebRTC's frame-by-frame `AudioResampler(\"s16\", \"mono\", 16000)` path",
        "- `vllm_pyav_float`: a 16 kHz float WAV generated with vLLM's whole-buffer `AudioResampler(format=\"fltp\", layout=\"mono\", rate=16000)` path",
        "",
        "The sweep then runs deterministic real multi-turn inference for both thinking-disabled and thinking-enabled profiles.",
        "",
        "## Summary",
        "",
        f"- Units: `{report['summary']['units_total']}`",
        f"- Trials per variant/profile: `{report['config']['trials']}`",
        f"- Profiles: `{', '.join(report['config']['profiles'])}`",
        "",
        "## Profiles",
        "",
    ]
    for profile_id, profile_summary in report["summary"]["profiles"].items():
        lines.append(f"### `{profile_id}`")
        lines.append("")
        lines.append("| Unit | Source48 | Pipecat | vLLM |")
        lines.append("| --- | --- | --- | --- |")
        for unit_summary in profile_summary["units"]:
            lines.append(
                f"| `{unit_summary['unit_id']}` | "
                f"`{unit_summary['source48_second_turn']}` | "
                f"`{unit_summary['pipecat_second_turn']}` | "
                f"`{unit_summary['vllm_second_turn']}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite_artifacts:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads(args.unit_manifest.read_text(encoding="utf-8"))
    all_units = _load_packaged_units(args.packaged_unit_manifest, max_units=None)
    units_by_id = {unit.unit_id: unit for unit in all_units}
    units: list[PackagedUnit] = []
    for item in selected:
        unit_id = f"{item['case_id']}__{item['family_id']}"
        if unit_id not in units_by_id:
            raise KeyError(f"Unit {unit_id} not found in packaged unit manifest")
        units.append(units_by_id[unit_id])

    report_units: list[dict[str, Any]] = []
    summary_profiles: dict[str, dict[str, Any]] = {
        profile.profile_id: {"units": []} for profile in PROFILES
    }

    for unit in units:
        unit_dir = args.out_dir / "samples" / unit.unit_id
        unit_dir.mkdir(parents=True, exist_ok=True)
        source_wav = _find_source_wav(unit, args.source_study_root)
        generated = generate_path_wavs(source_wav, unit_dir, basename="source48_original")
        variants = {
            "source48_original": source_wav,
            "pipecat_smallwebrtc": generated.pipecat_wav,
            "vllm_pyav_float": generated.vllm_wav,
        }

        unit_report = {
            "unit": {
                "unit_id": unit.unit_id,
                "case_id": unit.case.case_id,
                "topic": unit.case.topic,
                "family_id": unit.family.family_id,
                "bucket": unit.bucket,
                "audio_prompt": unit.case.audio_prompt,
                "final_prompt": unit.family.prompt_builder(unit.case),
                "expected_behavior": unit.family.expected_builder(unit.case).encoded(),
            },
            "generated_paths": {
                "source_wav": str(generated.source_wav),
                "pipecat_wav": str(generated.pipecat_wav),
                "vllm_wav": str(generated.vllm_wav),
                "metadata_json": str(generated.metadata_json),
            },
            "profiles": {},
        }

        for profile in PROFILES:
            profile_report = {"config": profile.__dict__, "variants": {}}
            for variant_id, sample_path in variants.items():
                first_results: list[dict[str, Any]] = []
                second_results: list[dict[str, Any]] = []
                for _trial_idx in range(args.trials):
                    first_payload = _build_audio_only_payload(unit.case, sample_path, args.model)
                    _apply_template_overrides(
                        first_payload,
                        enable_thinking=profile.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        first_payload,
                        temperature=profile.temperature,
                        top_k=None,
                        omit_top_k=profile.omit_top_k,
                        top_p=None,
                        omit_top_p=False,
                    )
                    first_result, _metadata, _meta_path = _run_payload(
                        base_url=args.base_url,
                        payload=first_payload,
                        probe_dir=None,
                    )
                    first_results.append(first_result)

                    second_payload = _build_real_multiturn_payload(
                        unit.case,
                        unit.family,
                        sample_path,
                        model=args.model,
                        first_turn_response=first_result,
                    )
                    _apply_template_overrides(
                        second_payload,
                        enable_thinking=profile.enable_thinking,
                    )
                    _apply_sampling_overrides(
                        second_payload,
                        temperature=profile.temperature,
                        top_k=None,
                        omit_top_k=profile.omit_top_k,
                        top_p=None,
                        omit_top_p=False,
                    )
                    second_result, _metadata2, _meta_path2 = _run_payload(
                        base_url=args.base_url,
                        payload=second_payload,
                        probe_dir=None,
                    )
                    second_results.append(second_result)

                profile_report["variants"][variant_id] = {
                    "sample_path": str(sample_path),
                    "first_turn": _summarize_first_turn(first_results),
                    "second_turn": _summarize_second_turn(unit, second_results),
                }

            unit_report["profiles"][profile.profile_id] = profile_report
            summary_profiles[profile.profile_id]["units"].append(
                {
                    "unit_id": unit.unit_id,
                    "source48_second_turn": profile_report["variants"]["source48_original"][
                        "second_turn"
                    ]["majority_behavior"],
                    "pipecat_second_turn": profile_report["variants"]["pipecat_smallwebrtc"][
                        "second_turn"
                    ]["majority_behavior"],
                    "vllm_second_turn": profile_report["variants"]["vllm_pyav_float"][
                        "second_turn"
                    ]["majority_behavior"],
                }
            )

        report_units.append(unit_report)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "unit_manifest": str(args.unit_manifest),
            "packaged_unit_manifest": str(args.packaged_unit_manifest),
            "source_study_root": str(args.source_study_root),
            "out_dir": str(args.out_dir),
            "trials": args.trials,
            "profiles": [profile.profile_id for profile in PROFILES],
        },
        "summary": {
            "units_total": len(units),
            "profiles": summary_profiles,
        },
        "units": report_units,
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
