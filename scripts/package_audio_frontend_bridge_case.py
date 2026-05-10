#!/usr/bin/env python3
"""Package a confirmed audio front-end bridge repro case."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_audio_path_fragility_study import (
    ALL_TEST_FAMILIES,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PILOT_CASES,
    _load_request_template,
    _normalize_behavior,
    _run_payload,
)
from run_audio_waveform_variant_study import (
    _apply_sampling_overrides,
    _apply_template_overrides,
    _build_audio_only_payload,
    _build_real_multiturn_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-frontend-bridge-repro-20260510"

VARIANT_FILES = {
    "source48_original": "source48_original.wav",
    "pipecat_smallwebrtc": "source48_original.pipecat-smallwebrtc.wav",
    "vllm_pyav_float": "source48_original.vllm-pyav-float.wav",
}


@dataclass(frozen=True)
class ReplayConfig:
    enable_thinking: bool = True
    temperature: float = 0.0
    omit_top_k: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _find_case(case_id: str):
    return next(case for case in PILOT_CASES if case.case_id == case_id)


def _find_family(family_id: str):
    return next(family for family in ALL_TEST_FAMILIES if family.family_id == family_id)


def _encode_tool_behavior(result: dict[str, Any], family, case) -> str:
    return _normalize_behavior(family, result).encoded()


def _majority(items: list[str]) -> tuple[str, int]:
    counter = Counter(items)
    return counter.most_common(1)[0]


def _copy_inputs(sample_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIANT_FILES.values():
        shutil.copy2(sample_dir / name, out_dir / name)
    for extra in ("sample_report.json", "source48_original.frontend-paths.json"):
        src = sample_dir / extra
        if src.is_file():
            shutil.copy2(src, out_dir / extra)


def _run_replay_for_variant(
    *,
    case,
    family,
    variant_id: str,
    sample_path: Path,
    base_url: str,
    model: str,
    trials: int,
    replay_cfg: ReplayConfig,
) -> dict[str, Any]:
    trial_results: list[dict[str, Any]] = []
    second_behaviors: list[str] = []
    first_behaviors: list[str] = []

    for _ in range(trials):
        first_payload = _build_audio_only_payload(case, sample_path, model)
        _apply_template_overrides(first_payload, enable_thinking=replay_cfg.enable_thinking)
        _apply_sampling_overrides(
            first_payload,
            temperature=replay_cfg.temperature,
            top_k=None,
            omit_top_k=replay_cfg.omit_top_k,
            top_p=None,
            omit_top_p=False,
        )
        first_result, _, _ = _run_payload(base_url=base_url, payload=first_payload, probe_dir=None)
        first_behavior = f"text:{(first_result.get('content') or '').strip()}"
        first_behaviors.append(first_behavior)

        second_payload = _build_real_multiturn_payload(
            case,
            family,
            sample_path,
            model=model,
            first_turn_response=first_result,
        )
        _apply_template_overrides(second_payload, enable_thinking=replay_cfg.enable_thinking)
        _apply_sampling_overrides(
            second_payload,
            temperature=replay_cfg.temperature,
            top_k=None,
            omit_top_k=replay_cfg.omit_top_k,
            top_p=None,
            omit_top_p=False,
        )
        second_result, _, _ = _run_payload(base_url=base_url, payload=second_payload, probe_dir=None)
        second_behavior = _encode_tool_behavior(second_result, family, case)
        second_behaviors.append(second_behavior)
        trial_results.append(
            {
                "first_turn": {
                    "content": first_result.get("content") or "",
                    "prompt_tokens": first_result["prompt_tokens"],
                },
                "second_turn": {
                    "behavior": second_behavior,
                    "content": second_result.get("content") or "",
                    "tool_calls": second_result.get("tool_calls") or [],
                    "prompt_tokens": second_result["prompt_tokens"],
                },
            }
        )

    first_majority, first_majority_count = _majority(first_behaviors)
    second_majority, second_majority_count = _majority(second_behaviors)
    expected = family.expected_builder(case).encoded()
    return {
        "variant_id": variant_id,
        "sample_path": str(sample_path),
        "trials": trial_results,
        "first_turn": {
            "behaviors": first_behaviors,
            "majority_behavior": first_majority,
            "majority_count": first_majority_count,
        },
        "second_turn": {
            "behaviors": second_behaviors,
            "majority_behavior": second_majority,
            "majority_count": second_majority_count,
            "expected_behavior": expected,
            "matches_expected_all_trials": all(v == expected for v in second_behaviors),
        },
    }


def _run_text_only_analog(
    *,
    case,
    family,
    first_turn_text: str,
    base_url: str,
    model: str,
    trials: int,
    replay_cfg: ReplayConfig,
) -> dict[str, Any]:
    payload = _load_request_template()
    payload["model"] = model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["messages"] = [
        payload["messages"][0],
        {"role": "assistant", "content": first_turn_text},
        {"role": "user", "content": family.prompt_builder(case)},
    ]
    _apply_template_overrides(payload, enable_thinking=replay_cfg.enable_thinking)
    _apply_sampling_overrides(
        payload,
        temperature=replay_cfg.temperature,
        top_k=None,
        omit_top_k=replay_cfg.omit_top_k,
        top_p=None,
        omit_top_p=False,
    )

    behaviors: list[str] = []
    results: list[dict[str, Any]] = []
    for _ in range(trials):
        result, _, _ = _run_payload(base_url=base_url, payload=payload, probe_dir=None)
        behavior = _encode_tool_behavior(result, family, case)
        behaviors.append(behavior)
        results.append(
            {
                "behavior": behavior,
                "content": result.get("content") or "",
                "tool_calls": result.get("tool_calls") or [],
                "prompt_tokens": result["prompt_tokens"],
            }
        )
    majority_behavior, majority_count = _majority(behaviors)
    expected = family.expected_builder(case).encoded()
    return {
        "assistant_text": first_turn_text,
        "trials": results,
        "majority_behavior": majority_behavior,
        "majority_count": majority_count,
        "expected_behavior": expected,
        "matches_expected_all_trials": all(v == expected for v in behaviors),
    }


def _write_readme(path: Path, bundle: dict[str, Any]) -> None:
    summary = bundle["summary"]
    lines = [
        "# Audio Front-End Bridge Repro",
        "",
        "This bundle packages a confirmed bridge repro candidate where:",
        "",
        "- the original `48 kHz` source succeeds",
        "- the `vLLM` whole-buffer PyAV path succeeds",
        "- the Pipecat SmallWebRTC frame-by-frame PyAV `s16` path fails",
        "",
        "## Case",
        "",
        f"- Sample id: `{bundle['sample']['sample_id']}`",
        f"- Transcript: `{bundle['sample']['transcript']}`",
        f"- Emotion: `{bundle['sample']['emotion']}`",
        f"- Speed: `{bundle['sample']['speed']}`",
        f"- Volume: `{bundle['sample']['volume']}`",
        f"- Case id: `{bundle['case']['case_id']}`",
        f"- Family id: `{bundle['family']['family_id']}`",
        "",
        "## Replay Summary",
        "",
        f"- Bridge hit confirmed: `{summary['bridge_hit_confirmed']}`",
        f"- `source48_original` second-turn majority: `{summary['source48_second_turn']}`",
        f"- `pipecat_smallwebrtc` second-turn majority: `{summary['pipecat_second_turn']}`",
        f"- `vllm_pyav_float` second-turn majority: `{summary['vllm_second_turn']}`",
        "",
        "## Text-Only Analog",
        "",
        "The exact first-turn assistant texts were replayed without audio in context.",
        f"All three text-only analogs matched expected on all trials: `{summary['text_only_all_match_expected']}`",
        "",
        "## Files",
        "",
        "- `source48_original.wav`",
        "- `source48_original.pipecat-smallwebrtc.wav`",
        "- `source48_original.vllm-pyav-float.wav`",
        "- `source48_original.frontend-paths.json`",
        "- `sample_report.json`",
        "- `replay_results.json`",
        "- `text_only_analog_results.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    case = _find_case(args.case_id)
    family = _find_family(args.family_id)
    sample_report = json.loads((args.sample_dir / "sample_report.json").read_text(encoding="utf-8"))

    out_dir = args.out_dir / sample_report["sample"]["sample_id"]
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_inputs(args.sample_dir, out_dir)

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

    summary = {
        "source48_second_turn": replay_results["source48_original"]["second_turn"]["majority_behavior"],
        "pipecat_second_turn": replay_results["pipecat_smallwebrtc"]["second_turn"]["majority_behavior"],
        "vllm_second_turn": replay_results["vllm_pyav_float"]["second_turn"]["majority_behavior"],
        "bridge_hit_confirmed": (
            replay_results["source48_original"]["second_turn"]["matches_expected_all_trials"]
            and replay_results["vllm_pyav_float"]["second_turn"]["matches_expected_all_trials"]
            and not replay_results["pipecat_smallwebrtc"]["second_turn"]["matches_expected_all_trials"]
        ),
        "text_only_all_match_expected": all(
            result["matches_expected_all_trials"] for result in text_only_results.values()
        ),
    }

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "sample": sample_report["sample"],
        "case": {"case_id": args.case_id, "audio_prompt": case.audio_prompt},
        "family": {"family_id": args.family_id, "prompt": family.prompt_builder(case)},
        "summary": summary,
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
        json.dumps(bundle, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(out_dir / "README.md", bundle)
    print(json.dumps(bundle["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
