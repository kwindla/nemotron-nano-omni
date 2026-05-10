#!/usr/bin/env python3
"""Package control units and strongest deterministic examples from waveform study runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS = ROOT / "artifacts" / "audio-waveform-variant-study-20260504" / "results.json"
ALT_RESULTS = (
    ROOT / "artifacts" / "audio-waveform-variant-study-temp0-no-topk-20260505" / "results.json"
)
OUT_DIR = ROOT / "artifacts" / "audio-waveform-variant-comparison-20260505"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _idx_units(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {unit["unit"]["unit_id"]: unit for unit in results["units"]}


def _semantic_flip_score(source: str, variant: str) -> int:
    source_kind = source.split(":", 1)[0]
    variant_kind = variant.split(":", 1)[0]
    return 1 if source_kind != variant_kind else 0


def _example_type(
    *,
    base_source: str,
    alt_source: str,
    base_variant: str,
    alt_variant: str,
) -> str:
    base_diff = base_variant != base_source
    alt_diff = alt_variant != alt_source
    source_changed = base_source != alt_source
    variant_changed = base_variant != alt_variant
    if base_diff and alt_diff and not source_changed and not variant_changed:
        return "stable_cross_run_fragility"
    if base_diff and alt_diff and source_changed and not variant_changed:
        return "source48_anchor_shift_with_stable_variant"
    if (not base_diff) and alt_diff and not source_changed:
        return "deterministic_run_revealed_fragility"
    if (not base_diff) and alt_diff and source_changed:
        return "source48_anchor_shift_revealed_fragility"
    if base_diff and alt_diff and variant_changed:
        return "cross_run_fragility_with_behavior_change"
    return "other"


def _unit_summary(unit: dict[str, Any]) -> dict[str, Any]:
    src = unit["variants"]["source48_original"]["majority_behavior"]
    diff = 0
    stoch = 0
    for variant_id, variant in unit["variants"].items():
        if variant_id == "source48_original":
            continue
        if variant["majority_behavior"] != src:
            diff += 1
        if variant["within_variant_stochastic"]:
            stoch += 1
    return {
        "source48_majority_behavior": src,
        "variants_differing_from_source48": diff,
        "stochastic_variants": stoch,
    }


def _build_unit_comparison(base_unit: dict[str, Any], alt_unit: dict[str, Any]) -> dict[str, Any]:
    base_summary = _unit_summary(base_unit)
    alt_summary = _unit_summary(alt_unit)
    return {
        "unit": base_unit["unit"],
        "base": base_summary,
        "alt": alt_summary,
        "delta": {
            "variants_differing_from_source48": (
                alt_summary["variants_differing_from_source48"]
                - base_summary["variants_differing_from_source48"]
            ),
            "stochastic_variants": (
                alt_summary["stochastic_variants"] - base_summary["stochastic_variants"]
            ),
            "source48_majority_changed": (
                base_summary["source48_majority_behavior"]
                != alt_summary["source48_majority_behavior"]
            ),
        },
    }


def _family_rollup(unit_comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "units": 0,
            "base_differing_variants": 0,
            "alt_differing_variants": 0,
            "base_stochastic_variants": 0,
            "alt_stochastic_variants": 0,
            "units_with_source48_majority_change": 0,
        }
    )
    for comp in unit_comparisons:
        family_id = comp["unit"]["family_id"]
        entry = per_family[family_id]
        entry["units"] += 1
        entry["base_differing_variants"] += comp["base"]["variants_differing_from_source48"]
        entry["alt_differing_variants"] += comp["alt"]["variants_differing_from_source48"]
        entry["base_stochastic_variants"] += comp["base"]["stochastic_variants"]
        entry["alt_stochastic_variants"] += comp["alt"]["stochastic_variants"]
        entry["units_with_source48_majority_change"] += int(
            comp["delta"]["source48_majority_changed"]
        )
    return dict(sorted(per_family.items()))


def _select_strong_examples(
    base_idx: dict[str, dict[str, Any]],
    alt_idx: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for unit_id, alt_unit in alt_idx.items():
        if alt_unit["unit"]["bucket"] == "stable_control":
            continue
        alt_source = alt_unit["variants"]["source48_original"]["majority_behavior"]
        base_source = base_idx[unit_id]["variants"]["source48_original"]["majority_behavior"]

        candidates = []
        for variant_id, alt_variant in alt_unit["variants"].items():
            if variant_id == "source48_original":
                continue
            if alt_variant["within_variant_stochastic"]:
                continue
            if alt_variant["majority_behavior"] == alt_source:
                continue
            base_variant = base_idx[unit_id]["variants"][variant_id]
            score = 0.0
            score += 10.0
            score += 5.0 * _semantic_flip_score(alt_source, alt_variant["majority_behavior"])
            score += 2.0 * int(not base_variant["within_variant_stochastic"])
            score += 2.0 * abs(
                alt_variant["expected_match_rate"]
                - alt_unit["variants"]["source48_original"]["expected_match_rate"]
            )
            score += 1.0 * abs(
                base_variant["expected_match_rate"]
                - base_idx[unit_id]["variants"]["source48_original"]["expected_match_rate"]
            )
            score += 1.0 * int(base_variant["majority_behavior"] != base_source)
            candidates.append(
                {
                    "score": score,
                    "variant_id": variant_id,
                    "alt_variant": alt_variant,
                    "base_variant": base_variant,
                }
            )

        candidates.sort(key=lambda item: (-item["score"], item["variant_id"]))
        for item in candidates[:2]:
            example_type = _example_type(
                base_source=base_source,
                alt_source=alt_source,
                base_variant=item["base_variant"]["majority_behavior"],
                alt_variant=item["alt_variant"]["majority_behavior"],
            )
            examples.append(
                {
                    "unit": alt_unit["unit"],
                    "variant_id": item["variant_id"],
                    "variant_group": item["alt_variant"]["variant_group"],
                    "variant_description": item["alt_variant"]["variant_description"],
                    "base_source48_majority": base_source,
                    "alt_source48_majority": alt_source,
                    "base_majority_behavior": item["base_variant"]["majority_behavior"],
                    "alt_majority_behavior": item["alt_variant"]["majority_behavior"],
                    "base_expected_match_rate": item["base_variant"]["expected_match_rate"],
                    "alt_expected_match_rate": item["alt_variant"]["expected_match_rate"],
                    "base_stochastic": item["base_variant"]["within_variant_stochastic"],
                    "alt_stochastic": item["alt_variant"]["within_variant_stochastic"],
                    "base_behavior_counts": item["base_variant"]["behavior_counts"],
                    "alt_behavior_counts": item["alt_variant"]["behavior_counts"],
                    "example_type": example_type,
                    "score": item["score"],
                }
            )
    examples.sort(
        key=lambda item: (
            item["unit"]["bucket"],
            -item["score"],
            item["unit"]["family_id"],
            item["unit"]["case_id"],
            item["variant_id"],
        )
    )
    return examples


def _write_readme(
    path: Path,
    *,
    controls: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    unit_comparisons: list[dict[str, Any]],
    family_rollup: dict[str, Any],
) -> None:
    lines = [
        "# Audio Waveform Example Package",
        "",
        "This package contains a compact set of stable controls and high-signal example cells",
        "drawn from the larger waveform-fragility studies.",
        "",
        "The goal is to show one thing clearly:",
        "",
        "> Semantically equivalent audio inputs can produce substantively different model",
        "> behavior because of very small waveform-level differences.",
        "",
        "## Stable Controls",
        "",
        f"- Stable control units packaged: `{len(controls)}`",
    ]
    for control in controls:
        lines.append(
            f"- `{control['unit']['unit_id']}`: source and all tested variants stayed stable "
            f"across repeated testing"
        )

    lines.extend(
        [
            "",
            "These controls are useful because they show the harness is not just noisy. Under the",
            "same overall methodology, some tasks remain perfectly stable while others do not.",
            "",
            "## Family Rollup",
            "",
            "| Family | Units | Base differing variants | Alt differing variants | Base stochastic variants | Alt stochastic variants | Source48 majority changed units |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family_id, roll in family_rollup.items():
        lines.append(
            f"| `{family_id}` | `{roll['units']}` | `{roll['base_differing_variants']}` | "
            f"`{roll['alt_differing_variants']}` | `{roll['base_stochastic_variants']}` | "
            f"`{roll['alt_stochastic_variants']}` | `{roll['units_with_source48_majority_change']}` |"
        )

        lines.extend(
        [
            "",
            "## High-Signal Fragile Examples",
            "",
            "The packaged example cells in `deterministic_examples.json` are high-signal cases where",
            "small waveform perturbations changed the model's majority behavior.",
            "",
            f"- Example cells packaged: `{len(examples)}`",
            "",
            "| Unit | Variant | Example type | Base source48 | Alt source48 | Base majority | Alt majority |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for example in examples:
        lines.append(
            f"| `{example['unit']['unit_id']}` | `{example['variant_id']}` | "
            f"`{example['example_type']}` | `{example['base_source48_majority']}` | `{example['alt_source48_majority']}` | "
            f"`{example['base_majority_behavior']}` | `{example['alt_majority_behavior']}` |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `stable_controls.json`",
            "  - stable control units",
            "- `deterministic_examples.json`",
            "  - strongest packaged example cells",
            "- `unit_comparisons.json`",
            "  - per-unit comparison data used during curation",
            "- `family_rollup.json`",
            "  - family-level rollup data used during curation",
            "",
            "## Intended Reading",
            "",
            "This package should be read as evidence of **waveform fragility**, not as a study of",
            "sampling temperature or decoding policy.",
            "",
            "The central question is:",
            "",
            "- if the semantic content is the same, but the waveform differs slightly, does the model",
            "  behave differently?",
            "",
            "The packaged examples show that the answer is yes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = _load(BASE_RESULTS)
    alt = _load(ALT_RESULTS)
    base_idx = _idx_units(base)
    alt_idx = _idx_units(alt)

    unit_comparisons = [
        _build_unit_comparison(base_idx[unit_id], alt_idx[unit_id])
        for unit_id in base_idx.keys()
    ]

    controls = [
        comp
        for comp in unit_comparisons
        if comp["unit"]["bucket"] == "stable_control"
    ]
    family_rollup = _family_rollup(unit_comparisons)
    examples = _select_strong_examples(base_idx, alt_idx)

    (OUT_DIR / "stable_controls.json").write_text(
        json.dumps(controls, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "unit_comparisons.json").write_text(
        json.dumps(unit_comparisons, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "family_rollup.json").write_text(
        json.dumps(family_rollup, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "deterministic_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        OUT_DIR / "README.md",
        controls=controls,
        examples=examples,
        unit_comparisons=unit_comparisons,
        family_rollup=family_rollup,
    )

    print(
        json.dumps(
            {
                "controls": len(controls),
                "unit_comparisons": len(unit_comparisons),
                "deterministic_examples": len(examples),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
