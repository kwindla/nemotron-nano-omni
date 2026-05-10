#!/usr/bin/env python3
"""Curate a high-signal golden subset from the full fragility screening results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_DIR = ROOT / "artifacts" / "audio-path-fragility-study-20260504"
DEFAULT_OUT_DIR = DEFAULT_STUDY_DIR / "golden-subset"

CONTROL_QUOTAS = {
    "text_json_topic": 3,
    "text_uppercase_topic": 3,
}

DETERMINISTIC_QUOTAS = {
    "tool_pwd": 2,
    "tool_git_branch": 2,
    "tool_top_level_file_count": 2,
    "text_letters_times_three_plus_one": 2,
    "text_first_and_last_letter_only": 2,
    "text_alphabetical_sort_topic": 1,
    "text_vowel_count_plus_constant": 1,
}

STOCHASTIC_QUOTAS = {
    "tool_git_branch": 1,
    "tool_top_level_file_count": 1,
    "text_letters_times_three_plus_one": 1,
    "text_vowel_count_plus_constant": 2,
    "text_alphabetical_sort_topic": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_family_index(results: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for case_report in results["cases"]:
        case_id = case_report["case"]["case_id"]
        for family_report in case_report["families"]:
            idx[(case_id, family_report["family_id"])] = {
                "case": case_report["case"],
                "family": family_report,
            }
    return idx


def _sort_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(unit["expected_rate_spread"]),
        -int(unit["cross_variant_disagreement"]),
        -int(unit["within_variant_stochastic"]),
        -int(unit["has_repeated_letters"]),
        -int(unit["topic_length"]),
        unit["case_id"],
        unit["family_id"],
    )


def _pick_units(
    *,
    candidates: list[dict[str, Any]],
    quota: int,
    used_case_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    unique_first = sorted(candidates, key=_sort_key)
    for unit in unique_first:
        if len(selected) >= quota:
            break
        if unit["case_id"] in used_case_ids:
            continue
        selected.append(unit)
        used_case_ids.add(unit["case_id"])

    if len(selected) >= quota:
        return selected

    selected_keys = {(unit["case_id"], unit["family_id"]) for unit in selected}
    for unit in unique_first:
        if len(selected) >= quota:
            break
        key = (unit["case_id"], unit["family_id"])
        if key in selected_keys:
            continue
        selected.append(unit)
        selected_keys.add(key)
        used_case_ids.add(unit["case_id"])

    return selected


def _select_by_family(
    *,
    units: list[dict[str, Any]],
    quotas: dict[str, int],
    used_case_ids: set[str],
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for family_id, quota in quotas.items():
        family_units = [unit for unit in units if unit["family_id"] == family_id]
        picked.extend(
            _pick_units(candidates=family_units, quota=quota, used_case_ids=used_case_ids)
        )
    return picked


def _decorate_units(
    units: list[dict[str, Any]],
    index: dict[tuple[str, str], dict[str, Any]],
    *,
    bucket: str,
    selection_reason: str,
) -> list[dict[str, Any]]:
    decorated = []
    for unit in units:
        lookup = index[(unit["case_id"], unit["family_id"])]
        family = lookup["family"]
        decorated.append(
            {
                "case_id": unit["case_id"],
                "family_id": unit["family_id"],
                "topic": unit["topic"],
                "bucket": bucket,
                "selection_reason": selection_reason,
                "family_group": unit["family_group"],
                "original_retention_label": unit["retention_label"],
                "original_retention_reason": unit["retention_reason"],
                "expected_rate_spread": unit["expected_rate_spread"],
                "cross_variant_disagreement": unit["cross_variant_disagreement"],
                "within_variant_stochastic": unit["within_variant_stochastic"],
                "majority_behaviors": unit["majority_behaviors"],
                "expected_behavior": family["expected_behavior"],
            }
        )
    return decorated


def _write_readme(
    path: Path,
    *,
    selected_units: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Golden Subset Curation",
        "",
        "This subset is intended for deeper repeated-trial reruns and external packaging.",
        "It keeps a balanced mix of stable controls, deterministic cross-variant fragility,",
        "and stochastic boundary cases from the full audio-path fragility study.",
        "",
        "## Summary",
        "",
        f"- Selected units: `{summary['selected_units_total']}`",
        f"- Stable controls: `{summary['bucket_counts'].get('stable_control', 0)}`",
        f"- Deterministic fragile: `{summary['bucket_counts'].get('deterministic_fragile', 0)}`",
        f"- Stochastic boundary: `{summary['bucket_counts'].get('stochastic_boundary', 0)}`",
        "",
        "## Family Counts",
        "",
    ]

    for family_id, count in sorted(summary["family_counts"].items()):
        lines.append(f"- `{family_id}`: `{count}`")

    lines.extend(
        [
            "",
            "## Selected Units",
            "",
            "| Bucket | Case | Topic | Family | Majority behaviors from screening |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for unit in selected_units:
        lines.append(
            f"| `{unit['bucket']}` | `{unit['case_id']}` | `{unit['topic']}` | "
            f"`{unit['family_id']}` | `{json.dumps(unit['majority_behaviors'], ensure_ascii=True)}` |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.out_dir} already exists; pass --overwrite to replace it")
        for child in args.out_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    retained = _load_json(args.study_dir / "retained_case_families.json")
    results = _load_json(args.study_dir / "results.json")
    index = _case_family_index(results)

    control_units = [unit for unit in retained if unit["retention_label"] == "stable_control"]
    deterministic_units = [
        unit
        for unit in retained
        if unit["retention_label"] == "fragile_cross_variant"
        and not unit["within_variant_stochastic"]
    ]
    stochastic_units = [
        unit for unit in retained if unit["retention_label"] == "fragile_stochastic"
    ]

    used_case_ids: set[str] = set()
    selected_controls = _select_by_family(
        units=control_units, quotas=CONTROL_QUOTAS, used_case_ids=used_case_ids
    )
    selected_deterministic = _select_by_family(
        units=deterministic_units, quotas=DETERMINISTIC_QUOTAS, used_case_ids=used_case_ids
    )
    selected_stochastic = _select_by_family(
        units=stochastic_units, quotas=STOCHASTIC_QUOTAS, used_case_ids=used_case_ids
    )

    selected_units = (
        _decorate_units(
            selected_controls,
            index,
            bucket="stable_control",
            selection_reason="high-confidence stable control family",
        )
        + _decorate_units(
            selected_deterministic,
            index,
            bucket="deterministic_fragile",
            selection_reason="clear cross-variant behavior split without within-variant noise",
        )
        + _decorate_units(
            selected_stochastic,
            index,
            bucket="stochastic_boundary",
            selection_reason="variant-sensitive behavior with within-variant stochasticity",
        )
    )

    family_counts = Counter(unit["family_id"] for unit in selected_units)
    bucket_counts = Counter(unit["bucket"] for unit in selected_units)
    summary = {
        "selected_units_total": len(selected_units),
        "family_counts": dict(sorted(family_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
    }

    (args.out_dir / "selected_units.json").write_text(
        json.dumps(selected_units, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "rerun_manifest.json").write_text(
        json.dumps(
            [
                {
                    "case_id": unit["case_id"],
                    "family_id": unit["family_id"],
                    "bucket": unit["bucket"],
                    "selection_reason": unit["selection_reason"],
                }
                for unit in selected_units
            ],
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", selected_units=selected_units, summary=summary)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
