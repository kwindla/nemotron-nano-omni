#!/usr/bin/env python3
"""Run the direct cache-state parity suite against a live vLLM server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_direct_vllm_fixture_parity import (
    DirectClient,
    SCENARIOS,
    compare_mode_outputs,
    run_mode,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nemotron_3_nano_omni"

SCENARIO_GROUPS = {
    "mixed20": ["mixed20"],
    "matrix": [
        "text-text",
        "text-tool-text",
        "audio-text",
        "audio-tool-text",
        "image-text",
        "image-tool-text",
    ],
    "publish-points": [
        "publish-plain-text",
        "publish-tool",
        "publish-multimodal",
        "publish-multimodal-tool",
    ],
    "stable": [
        "text-text",
        "text-tool-text",
        "audio-text",
        "audio-tool-text",
        "image-text",
        "image-tool-text",
        "publish-plain-text",
        "publish-tool",
        "publish-multimodal",
        "publish-multimodal-tool",
    ],
    "all": [
        "mixed20",
        "text-text",
        "text-tool-text",
        "audio-text",
        "audio-tool-text",
        "image-text",
        "image-tool-text",
        "publish-plain-text",
        "publish-tool",
        "publish-multimodal",
        "publish-multimodal-tool",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--group",
        choices=sorted(SCENARIO_GROUPS),
        default="stable",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "Optional vLLM conversation trace directory. When set, compare the "
            "actual rendered prompt traces for cached vs uncached requests."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = DirectClient(base_url=args.base_url, model=args.model)
    scenarios = SCENARIO_GROUPS[args.group]
    suite_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for scenario_name in scenarios:
        actions = SCENARIOS[scenario_name]
        cached = run_mode(
            client,
            actions=actions,
            scenario_name=scenario_name,
            cached=True,
            max_turns=None,
        )
        uncached = run_mode(
            client,
            actions=actions,
            scenario_name=scenario_name,
            cached=False,
            max_turns=None,
        )
        divergences = compare_mode_outputs(
            cached,
            uncached,
            trace_dir=args.trace_dir,
        )
        scenario_result = {
            "scenario": scenario_name,
            "cached_turn_count": len(cached["turns"]),
            "uncached_turn_count": len(uncached["turns"]),
            "divergences": divergences,
            "cached": cached,
            "uncached": uncached,
        }
        suite_results.append(scenario_result)
        if divergences:
            failures.append(
                {
                    "scenario": scenario_name,
                    "divergence": divergences[0],
                }
            )

    summary = {
        "group": args.group,
        "trace_dir": str(args.trace_dir) if args.trace_dir else None,
        "scenario_count": len(suite_results),
        "failures": failures,
        "results": suite_results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.summary_json)
    if failures:
        print(json.dumps(failures[0], indent=2))
        return 1
    print("no divergences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
