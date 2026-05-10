#!/usr/bin/env python3
"""Score first-turn responses semantically and aggregate by variant."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "first-turn-semantic-scoring-20260507"
RUBRIC_PATH = OUT_DIR / "RUBRIC.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nemotron_3_nano_omni"

RUNS = [
    {
        "run_id": "audio_only_thinking_disabled",
        "payload_mode": "audio_only",
        "enable_thinking": False,
        "results_path": ROOT
        / "artifacts"
        / "audio-waveform-variant-study-audio-only-20260505"
        / "results.json",
        "use_first_turn": False,
    },
    {
        "run_id": "audio_only_thinking_enabled",
        "payload_mode": "audio_only",
        "enable_thinking": True,
        "results_path": ROOT
        / "artifacts"
        / "audio-waveform-variant-study-audio-only-thinking-20260505"
        / "results.json",
        "use_first_turn": False,
    },
    {
        "run_id": "real_multiturn_thinking_disabled",
        "payload_mode": "real_multiturn",
        "enable_thinking": False,
        "results_path": ROOT
        / "artifacts"
        / "audio-waveform-variant-study-real-multiturn-20260506"
        / "results.json",
        "use_first_turn": True,
    },
    {
        "run_id": "real_multiturn_thinking_enabled",
        "payload_mode": "real_multiturn",
        "enable_thinking": True,
        "results_path": ROOT
        / "artifacts"
        / "audio-waveform-variant-study-real-multiturn-thinking-20260506"
        / "results.json",
        "use_first_turn": True,
    },
]


@dataclass(frozen=True)
class Example:
    run_id: str
    payload_mode: str
    enable_thinking: bool
    unit_id: str
    case_id: str
    topic: str
    family_id: str
    bucket: str
    variant_id: str
    trial_index: int
    user_prompt: str
    candidate_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-unique", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _strip_behavior_prefix(value: str) -> str:
    if value.startswith("text:"):
        return value[5:]
    if value.startswith("tool:"):
        return value[5:]
    return value


def _extract_examples() -> list[Example]:
    examples: list[Example] = []
    for run in RUNS:
        data = json.loads(run["results_path"].read_text())
        for unit in data["units"]:
            meta = unit["unit"]
            for variant_id, variant in unit["variants"].items():
                source = variant["first_turn"] if run["use_first_turn"] else variant
                for trial_index, behavior in enumerate(source["behaviors"]):
                    examples.append(
                        Example(
                            run_id=run["run_id"],
                            payload_mode=run["payload_mode"],
                            enable_thinking=bool(run["enable_thinking"]),
                            unit_id=meta["unit_id"],
                            case_id=meta["case_id"],
                            topic=meta["topic"],
                            family_id=meta["family_id"],
                            bucket=meta["bucket"],
                            variant_id=variant_id,
                            trial_index=trial_index,
                            user_prompt=meta["audio_prompt"],
                            candidate_response=_strip_behavior_prefix(behavior),
                        )
                    )
    return examples


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    return json.loads(match.group(0))


def _judge_prompt(rubric: str, expected_topic: str, user_prompt: str, candidate_response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": rubric},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "expected_topic": expected_topic,
                    "user_prompt": user_prompt,
                    "candidate_response": candidate_response,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        },
    ]


def _judge_one(
    *,
    base_url: str,
    model: str,
    rubric: str,
    expected_topic: str,
    user_prompt: str,
    candidate_response: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "messages": _judge_prompt(rubric, expected_topic, user_prompt, candidate_response),
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    parsed = _extract_json_object(content)
    parsed["_raw_judge_response"] = content
    return parsed


def _validate_label(obj: dict[str, Any]) -> None:
    topic_ok = {"topic_correct", "topic_ambiguous", "topic_wrong"}
    semantic_ok = {"faithful", "loosely_faithful", "semantic_error"}
    format_ok = {"format_ok", "format_minor_deviation", "format_bad"}
    overall_ok = {"semantically_equivalent", "unclear", "semantically_different"}
    if obj.get("topic_label") not in topic_ok:
        raise ValueError(f"Invalid topic_label: {obj}")
    if obj.get("semantic_label") not in semantic_ok:
        raise ValueError(f"Invalid semantic_label: {obj}")
    if obj.get("format_label") not in format_ok:
        raise ValueError(f"Invalid format_label: {obj}")
    if obj.get("overall_label") not in overall_ok:
        raise ValueError(f"Invalid overall_label: {obj}")


def _aggregate_examples(scored_examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_run_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_variant_global: dict[str, list[dict[str, Any]]] = {}
    for item in scored_examples:
        by_run_variant.setdefault((item["run_id"], item["variant_id"]), []).append(item)
        by_variant_global.setdefault(item["variant_id"], []).append(item)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        counts = {"semantically_equivalent": 0, "unclear": 0, "semantically_different": 0}
        for item in items:
            counts[item["overall_label"]] += 1
        return {
            "total": total,
            "counts": counts,
            "rates": {k: counts[k] / total for k in counts},
        }

    run_variant_summary = {
        f"{run_id}::{variant_id}": {
            "run_id": run_id,
            "variant_id": variant_id,
            **summarize(items),
        }
        for (run_id, variant_id), items in sorted(by_run_variant.items())
    }
    global_variant_summary = {
        variant_id: {"variant_id": variant_id, **summarize(items)}
        for variant_id, items in sorted(by_variant_global.items())
    }
    return {
        "run_variant_summary": run_variant_summary,
        "global_variant_summary": global_variant_summary,
    }


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and args.overwrite:
        for path in args.out_dir.iterdir():
            if path.is_file() and path.name != "RUBRIC.md":
                path.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    examples = _extract_examples()

    unique_keys: dict[tuple[str, str, str], None] = {}
    for ex in examples:
        unique_keys[(ex.topic, ex.user_prompt, ex.candidate_response)] = None
    unique_items = list(unique_keys.keys())
    if args.max_unique is not None:
        unique_items = unique_items[: args.max_unique]

    judged: dict[tuple[str, str, str], dict[str, Any]] = {}

    def work(item: tuple[str, str, str]) -> tuple[tuple[str, str, str], dict[str, Any]]:
        topic, prompt, response = item
        result = _judge_one(
            base_url=args.base_url,
            model=args.model,
            rubric=rubric,
            expected_topic=topic,
            user_prompt=prompt,
            candidate_response=response,
        )
        _validate_label(result)
        return item, result

    started_at = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(work, item): item for item in unique_items}
        for idx, future in enumerate(as_completed(futures), 1):
            item, result = future.result()
            judged[item] = result
            if idx % 100 == 0 or idx == len(unique_items):
                print(f"judged {idx}/{len(unique_items)} unique responses", flush=True)

    scored_examples: list[dict[str, Any]] = []
    missing = 0
    for ex in examples:
        key = (ex.topic, ex.user_prompt, ex.candidate_response)
        result = judged.get(key)
        if result is None:
            missing += 1
            continue
        scored_examples.append(
            {
                "run_id": ex.run_id,
                "payload_mode": ex.payload_mode,
                "enable_thinking": ex.enable_thinking,
                "unit_id": ex.unit_id,
                "case_id": ex.case_id,
                "topic": ex.topic,
                "family_id": ex.family_id,
                "bucket": ex.bucket,
                "variant_id": ex.variant_id,
                "trial_index": ex.trial_index,
                "user_prompt": ex.user_prompt,
                "candidate_response": ex.candidate_response,
                "topic_label": result["topic_label"],
                "semantic_label": result["semantic_label"],
                "format_label": result["format_label"],
                "overall_label": result["overall_label"],
                "short_rationale": result["short_rationale"],
            }
        )

    aggregates = _aggregate_examples(scored_examples)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "examples_total": len(examples),
        "unique_responses_total": len(unique_keys),
        "unique_responses_judged": len(judged),
        "examples_scored": len(scored_examples),
        "examples_missing_due_to_max_unique": missing,
    }

    unique_judgments = [
        {
            "topic": topic,
            "user_prompt": user_prompt,
            "candidate_response": candidate_response,
            **result,
        }
        for (topic, user_prompt, candidate_response), result in sorted(judged.items())
    ]

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "unique_judgments.json").write_text(
        json.dumps(unique_judgments, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "scored_examples.json").write_text(
        json.dumps(scored_examples, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "aggregates.json").write_text(
        json.dumps(aggregates, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
