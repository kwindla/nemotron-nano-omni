#!/usr/bin/env python3
"""Validate and package a set of audio front-end example cases."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from package_audio_frontend_bridge_case import (
    DEFAULT_OUT_DIR as DEFAULT_BRIDGE_OUT_DIR,
    VARIANT_FILES,
    ReplayConfig,
    _copy_inputs,
    _find_case,
    _find_family,
    _run_replay_for_variant,
    _run_text_only_analog,
)
from run_audio_waveform_variant_study import _normalize_audio_only_behavior

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = ROOT / "artifacts" / "mountain-frontend-search-20260509" / "samples"
DEFAULT_MANIFEST = ROOT / "artifacts" / "audio-frontend-example-set-20260510" / "examples_manifest.json"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-frontend-example-set-20260510"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="nemotron_3_nano_omni")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pattern_tuple(replay_results: dict[str, Any]) -> tuple[bool, bool, bool]:
    return (
        replay_results["source48_original"]["second_turn"]["matches_expected_all_trials"],
        replay_results["pipecat_smallwebrtc"]["second_turn"]["matches_expected_all_trials"],
        replay_results["vllm_pyav_float"]["second_turn"]["matches_expected_all_trials"],
    )


def _expected_tuple(label: str) -> tuple[bool, bool, bool]:
    if label == "bridge":
        return (True, False, True)
    if label == "rescue":
        return (False, True, False)
    raise ValueError(f"Unsupported expected pattern label: {label}")


def _write_readme(path: Path, bundle: dict[str, Any]) -> None:
    lines = [
        "# Audio Front-End Example Set",
        "",
        "This bundle packages five validated examples from the `096-mountain` search:",
        "",
        "- one confirmed bridge hit",
        "- four confirmed rescue cases",
        "",
        "Each case was checked with:",
        "",
        "- repeated deterministic replay of the exact three WAVs",
        "- text-only analog replay using the observed first-turn assistant texts",
        "",
        "## Summary",
        "",
        f"- Examples requested: `{bundle['summary']['examples_requested']}`",
        f"- Examples passing sanity check: `{bundle['summary']['examples_passed']}`",
        f"- Bridge examples passing: `{bundle['summary']['bridge_examples_passed']}`",
        f"- Rescue examples passing: `{bundle['summary']['rescue_examples_passed']}`",
        "",
        "## Examples",
        "",
    ]
    for example in bundle["examples"]:
        lines.extend(
            [
                f"### `{example['sample']['sample_id']}`",
                "",
                f"- expected pattern: `{example['expected_pattern']}`",
                f"- observed pattern: `{example['observed_pattern']}`",
                f"- sanity check passed: `{example['sanity_check_passed']}`",
                f"- transcript: `{example['sample']['transcript']}`",
                f"- emotion/profile: `{example['sample']['emotion']}` / `{example['sample']['profile_id']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle_examples: list[dict[str, Any]] = []
    bridge_examples_passed = 0
    rescue_examples_passed = 0

    for item in manifest:
        sample_id = item["sample_id"]
        sample_dir = args.search_root / sample_id
        sample_report = json.loads((sample_dir / "sample_report.json").read_text(encoding="utf-8"))
        case = _find_case(item["case_id"])
        family = _find_family(item["family_id"])

        out_dir = args.out_dir / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_inputs(sample_dir, out_dir)

        replay_cfg = ReplayConfig()
        replay_results: dict[str, Any] = {}
        for variant_id, filename in VARIANT_FILES.items():
            replay_results[variant_id] = _run_replay_for_variant(
                case=case,
                family=family,
                variant_id=variant_id,
                sample_path=out_dir / filename,
                base_url=args.base_url,
                model=args.model,
                trials=args.trials,
                replay_cfg=replay_cfg,
            )

        text_only_results: dict[str, Any] = {}
        for variant_id, variant_report in sample_report["variants"].items():
            text_only_results[variant_id] = _run_text_only_analog(
                case=case,
                family=family,
                first_turn_text=variant_report["first_turn"]["content"],
                base_url=args.base_url,
                model=args.model,
                trials=args.trials,
                replay_cfg=replay_cfg,
            )

        observed_tuple = _pattern_tuple(replay_results)
        expected_tuple = _expected_tuple(item["expected_pattern"])
        text_only_all_match = all(
            result["matches_expected_all_trials"] for result in text_only_results.values()
        )
        sanity_passed = observed_tuple == expected_tuple and text_only_all_match

        example_summary = {
            "sample": sample_report["sample"],
            "case": {"case_id": item["case_id"], "audio_prompt": case.audio_prompt},
            "family": {"family_id": item["family_id"], "prompt": family.prompt_builder(case)},
            "expected_pattern": item["expected_pattern"],
            "observed_pattern": {
                "source48": observed_tuple[0],
                "pipecat": observed_tuple[1],
                "vllm": observed_tuple[2],
            },
            "sanity_check_passed": sanity_passed,
            "text_only_all_match_expected": text_only_all_match,
        }

        (out_dir / "replay_results.json").write_text(
            json.dumps(replay_results, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "text_only_analog_results.json").write_text(
            json.dumps(text_only_results, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "bundle_summary.json").write_text(
            json.dumps(example_summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        bundle_examples.append(example_summary)
        if sanity_passed and item["expected_pattern"] == "bridge":
            bridge_examples_passed += 1
        if sanity_passed and item["expected_pattern"] == "rescue":
            rescue_examples_passed += 1

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "summary": {
            "examples_requested": len(manifest),
            "examples_passed": sum(1 for example in bundle_examples if example["sanity_check_passed"]),
            "bridge_examples_passed": bridge_examples_passed,
            "rescue_examples_passed": rescue_examples_passed,
        },
        "examples": bundle_examples,
    }

    (args.out_dir / "bundle_summary.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", bundle)
    print(json.dumps(bundle["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
