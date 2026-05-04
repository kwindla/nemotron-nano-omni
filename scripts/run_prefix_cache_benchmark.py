#!/usr/bin/env python3
"""Run alternating conversation-cache benchmarks and summarize TTFT behavior."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATFORM = os.getenv("NEMOTRON_PLATFORM") or "dgx_spark"
START_BOT_SCRIPT = ROOT / "scripts" / "start_bot.sh"
REGRESSION_SCRIPT = ROOT / "scripts" / "run_mixed_rtvi_regression.py"
PIPECAT_PYTHON = ROOT / ".venv-pipecat" / "bin" / "python"

TTFB_PATTERN = re.compile(r"NemotronOmniAudioLLMService#\d+ TTFB: ([0-9.]+)s")
RUNNING_BASH_RE = re.compile(r"running bash tool in .*: (?P<command>'.*')$")
TOOL_RESULT_RE = re.compile(r"model-facing bash tool result for .*: (?P<json>\{.*\})$")
FORBIDDEN_PATTERNS: dict[str, str] = {
    "409_conflict": r"409 Conflict",
    "already_generating": r"already generating",
    "conversation_cache_miss": r"ConversationCacheMissError|conversation cache miss",
    "attach_skipped": r"attach skipped",
    "traceback": r"Traceback \(most recent call last\)",
    "error_frame": r"audio completion failed|RTVI error|error-response",
}


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    conversation_cache_enabled: bool
    expect_cache_attach: str


CONFIGS = [
    ConfigSpec(
        name="conversation_cache_enabled",
        conversation_cache_enabled=True,
        expect_cache_attach="always",
    ),
    ConfigSpec(
        name="conversation_cache_disabled",
        conversation_cache_enabled=False,
        expect_cache_attach="never",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=int,
        default=10,
        help="Number of enabled/disabled benchmark pairs to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional benchmark output directory. Defaults to benchmarks/<timestamp>.",
    )
    parser.add_argument(
        "--platform",
        choices=("rtx5090", "dgx_spark"),
        default=DEFAULT_PLATFORM,
        help=(
            "Platform env profile to source. Defaults to NEMOTRON_PLATFORM, "
            "or dgx_spark when unset."
        ),
    )
    parser.add_argument(
        "--env-script",
        type=Path,
        help="Optional explicit env.sh path. Overrides --platform.",
    )
    parser.add_argument(
        "--vllm-ready-timeout-secs",
        type=float,
        default=60.0,
        help="How long to wait for the already-running standard vLLM server.",
    )
    parser.add_argument(
        "--turn-pause-secs",
        type=float,
        default=3.0,
        help=(
            "Pause between turns passed to the mixed RTVI regression. This "
            "needs to be long enough for the output transport to forward the "
            "prior assistant turn into the shared Pipecat context."
        ),
    )
    return parser.parse_args()


def platform_env_script(platform: str) -> Path:
    return ROOT / "platforms" / platform / "config" / "env.sh"


def load_platform_env(env_script: Path) -> dict[str, str]:
    command = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && env -0"
    output = subprocess.check_output(["bash", "-lc", command], cwd=ROOT)
    env = os.environ.copy()
    for item in output.split(b"\0"):
        if not item:
            continue
        key, _, value = item.partition(b"=")
        env[key.decode("utf-8")] = value.decode("utf-8")
    return env


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    stdout_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is None:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            check=check,
            capture_output=True,
        )

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            check=check,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def read_new_text(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        return handle.read()


def wait_for_port_free(port: int, *, timeout_secs: float) -> bool:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return True
        time.sleep(0.5)
    return False


def listener_pids(port: int) -> list[int]:
    output = subprocess.check_output(
        ["bash", "-lc", f"ss -H -ltnp '( sport = :{port} )' 2>/dev/null || true"],
        cwd=ROOT,
        text=True,
    )
    pids = []
    for match in re.finditer(r"pid=(\d+)", output):
        pids.append(int(match.group(1)))
    return sorted(set(pids))


def stop_bot(bot_env: dict[str, str], *, label: str) -> None:
    run_command(
        [str(START_BOT_SCRIPT), "stop"],
        env=bot_env,
        stdout_path=None,
        check=False,
    )
    if wait_for_port_free(7860, timeout_secs=5):
        return
    for pid in listener_pids(7860):
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            continue
    if not wait_for_port_free(7860, timeout_secs=15):
        raise RuntimeError(f"{label}: bot port 7860 stayed busy after stop")


def ensure_vllm_ready(env: dict[str, str], *, timeout_secs: float) -> None:
    models_url = f"{env['NEMOTRON_VLLM_BASE_URL']}/models"
    deadline = time.time() + timeout_secs
    last_error = "unknown error"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if any(
                item.get("id") == env["NEMOTRON_VLLM_MODEL"]
                for item in payload.get("data", [])
            ):
                return
            last_error = f"model {env['NEMOTRON_VLLM_MODEL']} not present in /models"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(
        f"Timed out after {timeout_secs}s waiting for standard vLLM at {models_url}: "
        f"{last_error}"
    )


def run_pair_member(
    *,
    base_env: dict[str, str],
    bot_pid_path: Path,
    out_root: Path,
    pair_index: int,
    config: ConfigSpec,
    turn_pause_secs: float,
) -> dict[str, Any]:
    run_dir = out_root / config.name / f"pair{pair_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shared_vllm_log = Path(base_env["NEMOTRON_VLLM_LOG"])
    vllm_log_offset = file_size(shared_vllm_log)

    bot_env = base_env.copy()
    bot_env.update(
        {
            "NEMOTRON_LOG_DIR": str(run_dir),
            "NEMOTRON_BOT_PID": str(bot_pid_path),
            "NEMOTRON_BOT_STDOUT_LOG": str(run_dir / "bot.stdout.log"),
            "NEMOTRON_OMNI_LOG": str(run_dir / "bot.log"),
            "NEMOTRON_OMNI_BASE_URL": base_env["NEMOTRON_VLLM_BASE_URL"],
            "NEMOTRON_OMNI_ENABLE_CONVERSATION_CACHE": (
                "1" if config.conversation_cache_enabled else "0"
            ),
            "NEMOTRON_OMNI_CONVERSATION_ID": "",
        }
    )

    metadata = {
        "pair_index": pair_index,
        "config": config.name,
        "conversation_cache_enabled": config.conversation_cache_enabled,
        "expect_cache_attach": config.expect_cache_attach,
        "shared_vllm_log": str(shared_vllm_log),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    stop_bot(bot_env, label=f"{config.name} pair {pair_index:02d} pre-stop")
    run_command(
        [str(START_BOT_SCRIPT), "start"],
        env=bot_env,
        stdout_path=run_dir / "start_bot.log",
    )
    time.sleep(2)

    regression_command = [
        str(PIPECAT_PYTHON),
        str(REGRESSION_SCRIPT),
        "--bot-log",
        str(run_dir / "bot.log"),
        "--vllm-log",
        str(shared_vllm_log),
        "--expect-cache-attach",
        config.expect_cache_attach,
        "--summary-json",
        str(run_dir / "summary.json"),
        "--turn-pause-secs",
        str(turn_pause_secs),
        "--allow-response-mismatches",
    ]
    completed = run_command(
        regression_command,
        env=base_env,
        stdout_path=run_dir / "regression.log",
        check=False,
    )

    vllm_slice = read_new_text(shared_vllm_log, vllm_log_offset)
    (run_dir / "vllm.log").write_text(vllm_slice, encoding="utf-8")
    stop_bot(bot_env, label=f"{config.name} pair {pair_index:02d} post-stop")

    finished_at = datetime.now(UTC).isoformat()
    status_payload = {
        **metadata,
        "finished_at": finished_at,
        "returncode": completed.returncode,
    }
    (run_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "run_dir": run_dir,
        "returncode": completed.returncode,
        "config": config.name,
        "pair_index": pair_index,
    }


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[max(0, rank - 1)]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def extract_ttft_series(bot_log_path: Path) -> list[float]:
    text = bot_log_path.read_text(encoding="utf-8", errors="replace")
    return [float(match.group(1)) for match in TTFB_PATTERN.finditer(text)]


def scan_forbidden_patterns(*paths: Path) -> dict[str, int]:
    counts = {name: 0 for name in FORBIDDEN_PATTERNS}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            counts[name] += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return counts


def extract_tool_commands(bot_log_path: Path) -> list[str]:
    text = bot_log_path.read_text(encoding="utf-8", errors="replace")
    commands: list[str] = []
    for line in text.splitlines():
        match = RUNNING_BASH_RE.search(line)
        if not match:
            continue
        try:
            commands.append(str(ast.literal_eval(match.group("command"))))
        except (ValueError, SyntaxError):
            commands.append(match.group("command").strip("'"))
    return commands


def extract_tool_results(bot_log_path: Path) -> list[dict[str, Any]]:
    text = bot_log_path.read_text(encoding="utf-8", errors="replace")
    results: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = TOOL_RESULT_RE.search(line)
        if not match:
            continue
        try:
            results.append(json.loads(match.group("json")))
        except json.JSONDecodeError:
            continue
    return results


def build_progression_svg(
    *,
    run_series: list[tuple[str, list[float]]],
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    if not run_series:
        return

    max_turns = max(len(values) for _, values in run_series)
    max_value = max(max(values) for _, values in run_series if values)
    width = 1100
    height = 720
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(index: int) -> float:
        if max_turns <= 1:
            return margin_left
        return margin_left + ((index - 1) / (max_turns - 1)) * plot_width

    def y_pos(value: float) -> float:
        scale_max = max_value * 1.1 if max_value else 1.0
        return margin_top + plot_height - (value / scale_max) * plot_height

    median_series: list[float] = []
    for turn_index in range(1, max_turns + 1):
        turn_values = [
            values[turn_index - 1]
            for _, values in run_series
            if len(values) >= turn_index
        ]
        median_series.append(median(turn_values))

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="90" y="32" font-family="monospace" font-size="24" fill="#111111">{title}</text>',
        f'<text x="90" y="52" font-family="monospace" font-size="12" fill="#444444">{subtitle}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111111" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111111" stroke-width="1"/>',
    ]

    y_ticks = 6
    scale_max = max_value * 1.1 if max_value else 1.0
    for tick in range(y_ticks + 1):
        value = scale_max * tick / y_ticks
        y = y_pos(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#444444">{value:.2f}s</text>'
        )

    for turn_index in range(1, max_turns + 1):
        x = x_pos(turn_index)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="monospace" font-size="11" fill="#444444">{turn_index}</text>'
        )

    palette = [
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#c6dbef",
        "#9e9ac8",
        "#807dba",
        "#6a51a3",
        "#54278f",
    ]
    for index, (label, values) in enumerate(run_series):
        points = " ".join(
            f"{x_pos(turn_index + 1):.2f},{y_pos(value):.2f}"
            for turn_index, value in enumerate(values)
        )
        color = palette[index % len(palette)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}" opacity="0.8"/>'
        )
        if values:
            lines.append(
                f'<text x="{x_pos(len(values)) + 6:.2f}" y="{y_pos(values[-1]) + 4:.2f}" font-family="monospace" font-size="10" fill="{color}">{label}</text>'
            )

    median_points = " ".join(
        f"{x_pos(turn_index + 1):.2f},{y_pos(value):.2f}"
        for turn_index, value in enumerate(median_series)
    )
    lines.append(
        f'<polyline fill="none" stroke="#d62728" stroke-width="3" points="{median_points}"/>'
    )
    lines.append(
        '<text x="900" y="88" font-family="monospace" font-size="12" fill="#d62728">median</text>'
    )
    lines.append(
        '<text x="90" y="680" font-family="monospace" font-size="12" fill="#444444">x-axis: turn number</text>'
    )
    lines.append(
        '<text x="20" y="360" transform="rotate(-90 20 360)" font-family="monospace" font-size="12" fill="#444444">TTFT seconds</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_runs(out_root: Path, *, pairs: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "pairs": pairs,
        "generated_at": datetime.now(UTC).isoformat(),
        "configs": {},
    }
    run_series_by_config: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    progression_rows_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for config in CONFIGS:
        config_dir = out_root / config.name
        run_dirs = sorted(config_dir.glob("pair*"))
        ttft_values: list[float] = []
        run_summaries: list[dict[str, Any]] = []
        action_variants: dict[str, set[str]] = defaultdict(set)
        aggregate_forbidden = {name: 0 for name in FORBIDDEN_PATTERNS}
        failed_runs = 0
        successful_runs = 0
        duplicate_tool_calls = 0
        failed_tool_calls = 0
        response_mismatch_count = 0
        response_mismatch_labels: dict[str, set[str]] = defaultdict(set)
        garbled_response_labels: dict[str, set[str]] = defaultdict(set)
        anomalous_runs: list[dict[str, Any]] = []
        failed_run_details: list[dict[str, Any]] = []

        for run_dir in run_dirs:
            summary_path = run_dir / "summary.json"
            status_path = run_dir / "status.json"
            bot_log = run_dir / "bot.log"
            vllm_log = run_dir / "vllm.log"

            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists()
                else {}
            )
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
            if status.get("returncode", 1) != 0 or not summary:
                failed_runs += 1
                failure_reason = ""
                regression_log = run_dir / "regression.log"
                if regression_log.exists():
                    lines = [
                        line.strip()
                        for line in regression_log.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()
                        if line.strip()
                    ]
                    failure_reason = lines[-1] if lines else ""
                failed_run_details.append(
                    {
                        "run_dir": str(run_dir.relative_to(out_root)),
                        "returncode": status.get("returncode"),
                        "reason": failure_reason,
                    }
                )
                ttft_series = extract_ttft_series(bot_log) if bot_log.exists() else []
                forbidden_counts = scan_forbidden_patterns(bot_log, vllm_log)
                run_summaries.append(
                    {
                        "run_dir": str(run_dir.relative_to(out_root)),
                        "returncode": status.get("returncode"),
                        "ttft_count": len(ttft_series),
                        "conversation_id": summary.get("conversation_id"),
                        "stats": summary.get("stats", {}),
                        "forbidden_counts": forbidden_counts,
                        "tool_command_count": None,
                        "expected_tool_calls": None,
                        "adjacent_duplicate_tool_calls": None,
                        "failed_tool_calls": None,
                    }
                )
                continue

            successful_runs += 1
            ttft_series = extract_ttft_series(bot_log) if bot_log.exists() else []
            ttft_values.extend(ttft_series)
            forbidden_counts = scan_forbidden_patterns(bot_log, vllm_log)
            response_mismatches = summary.get("response_mismatches", [])
            response_mismatch_count += len(response_mismatches)
            for mismatch in response_mismatches:
                label = mismatch.get("label")
                response = mismatch.get("response")
                if isinstance(label, str) and isinstance(response, str):
                    response_mismatch_labels[label].add(response)
            for key, value in forbidden_counts.items():
                aggregate_forbidden[key] += value

            if ttft_series:
                run_series_by_config[config.name].append((run_dir.name, ttft_series))
                for turn_index, value in enumerate(ttft_series, start=1):
                    progression_rows_by_config[config.name].append(
                        {
                            "run": run_dir.name,
                            "turn": turn_index,
                            "ttft_secs": value,
                        }
                    )

            for item in summary.get("completed_responses", []):
                action_variants[item["label"]].add(item["response"].strip())
                if "\n\n" in item["response"] or item["response"].strip() != item["response"]:
                    garbled_response_labels[item["label"]].add(item["response"])

            tool_commands = extract_tool_commands(bot_log) if bot_log.exists() else []
            tool_results = extract_tool_results(bot_log) if bot_log.exists() else []
            expected_tool_calls = sum(
                1
                for item in summary.get("completed_responses", [])
                if "-tool-" in item.get("label", "")
            )
            adjacent_duplicates = sum(
                1
                for left, right in zip(tool_commands, tool_commands[1:], strict=False)
                if left == right
            )
            unsuccessful_tool_results = [
                result
                for result in tool_results
                if not result.get("ok") or result.get("status") != "success"
            ]
            duplicate_tool_calls += max(0, len(tool_commands) - expected_tool_calls)
            duplicate_tool_calls += adjacent_duplicates
            failed_tool_calls += len(unsuccessful_tool_results)

            anomalies: list[str] = []
            if len(tool_commands) != expected_tool_calls:
                anomalies.append(
                    f"tool_command_count={len(tool_commands)} expected={expected_tool_calls}"
                )
            if response_mismatches:
                anomalies.append(f"response_mismatches={len(response_mismatches)}")
            if adjacent_duplicates:
                anomalies.append(f"adjacent_duplicate_tool_calls={adjacent_duplicates}")
            if unsuccessful_tool_results:
                anomalies.append(f"failed_tool_calls={len(unsuccessful_tool_results)}")
            if any(forbidden_counts.values()):
                anomalies.append("forbidden_log_patterns")
            if anomalies:
                anomalous_runs.append(
                    {
                        "run_dir": str(run_dir.relative_to(out_root)),
                        "anomalies": anomalies,
                        "tool_commands": tool_commands,
                        "tool_results": unsuccessful_tool_results,
                    }
                )

            run_summaries.append(
                {
                    "run_dir": str(run_dir.relative_to(out_root)),
                    "returncode": status.get("returncode"),
                    "ttft_count": len(ttft_series),
                    "conversation_id": summary.get("conversation_id"),
                    "stats": summary.get("stats", {}),
                    "response_mismatches": response_mismatches,
                    "forbidden_counts": forbidden_counts,
                    "tool_command_count": len(tool_commands),
                    "expected_tool_calls": expected_tool_calls,
                    "adjacent_duplicate_tool_calls": adjacent_duplicates,
                    "failed_tool_calls": len(unsuccessful_tool_results),
                }
            )

        config_report: dict[str, Any] = {
            "run_count": len(run_dirs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "ttft_count": len(ttft_values),
            "forbidden_counts": aggregate_forbidden,
            "duplicate_tool_calls": duplicate_tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "response_mismatch_count": response_mismatch_count,
            "response_mismatch_labels": {
                label: sorted(values)
                for label, values in sorted(response_mismatch_labels.items())
            },
            "garbled_response_labels": {
                label: sorted(values)
                for label, values in sorted(garbled_response_labels.items())
            },
            "anomalous_runs": anomalous_runs,
            "failed_run_details": failed_run_details,
            "runs": run_summaries,
            "response_variants": {
                label: sorted(values)
                for label, values in sorted(action_variants.items())
            },
        }
        if ttft_values:
            config_report["ttft_median_secs"] = round(median(ttft_values), 6)
            config_report["ttft_p95_secs"] = round(percentile_95(ttft_values), 6)
            config_report["ttft_min_secs"] = round(min(ttft_values), 6)
            config_report["ttft_max_secs"] = round(max(ttft_values), 6)
        report["configs"][config.name] = config_report

    artifacts: dict[str, str] = {}
    for config in CONFIGS:
        config_rows = progression_rows_by_config[config.name]
        csv_path = out_root / f"{config.name}_ttft_by_turn.csv"
        csv_path.write_text(
            "run,turn,ttft_secs\n"
            + "".join(
                f"{row['run']},{row['turn']},{row['ttft_secs']}\n"
                for row in config_rows
            ),
            encoding="utf-8",
        )
        svg_path = out_root / f"{config.name}_ttft_by_turn.svg"
        build_progression_svg(
            run_series=run_series_by_config[config.name],
            output_path=svg_path,
            title=f"{config.name} TTFT By Turn",
            subtitle=f"20-turn mixed RTVI regression, {pairs} runs",
        )
        artifacts[f"{config.name}_ttft_csv"] = str(csv_path.relative_to(out_root))
        artifacts[f"{config.name}_ttft_svg"] = str(svg_path.relative_to(out_root))
    report["artifacts"] = artifacts
    return report


def write_report(out_root: Path, report: dict[str, Any]) -> None:
    (out_root / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    enabled = report["configs"].get("conversation_cache_enabled", {})
    disabled = report["configs"].get("conversation_cache_disabled", {})
    lines = [
        "# Conversation Cache Benchmark",
        "",
        f"- pairs: {report['pairs']}",
        f"- generated_at: {report['generated_at']}",
        "",
        "## Run Counts",
        "",
        f"- conversation_cache_enabled successful_runs: {enabled.get('successful_runs', 'n/a')}",
        f"- conversation_cache_enabled failed_runs: {enabled.get('failed_runs', 'n/a')}",
        f"- conversation_cache_disabled successful_runs: {disabled.get('successful_runs', 'n/a')}",
        f"- conversation_cache_disabled failed_runs: {disabled.get('failed_runs', 'n/a')}",
        "",
        "## TTFT",
        "",
        f"- conversation_cache_enabled median: {enabled.get('ttft_median_secs', 'n/a')}s",
        f"- conversation_cache_enabled p95: {enabled.get('ttft_p95_secs', 'n/a')}s",
        f"- conversation_cache_disabled median: {disabled.get('ttft_median_secs', 'n/a')}s",
        f"- conversation_cache_disabled p95: {disabled.get('ttft_p95_secs', 'n/a')}s",
        "",
        "## Forbidden Pattern Counts",
        "",
        f"- conversation_cache_enabled: {json.dumps(enabled.get('forbidden_counts', {}), sort_keys=True)}",
        f"- conversation_cache_disabled: {json.dumps(disabled.get('forbidden_counts', {}), sort_keys=True)}",
        "",
        "## Tool And Response Anomalies",
        "",
        f"- conversation_cache_enabled duplicate_tool_calls: {enabled.get('duplicate_tool_calls', 'n/a')}",
        f"- conversation_cache_enabled failed_tool_calls: {enabled.get('failed_tool_calls', 'n/a')}",
        f"- conversation_cache_enabled response_mismatch_count: {enabled.get('response_mismatch_count', 'n/a')}",
        f"- conversation_cache_disabled duplicate_tool_calls: {disabled.get('duplicate_tool_calls', 'n/a')}",
        f"- conversation_cache_disabled failed_tool_calls: {disabled.get('failed_tool_calls', 'n/a')}",
        f"- conversation_cache_disabled response_mismatch_count: {disabled.get('response_mismatch_count', 'n/a')}",
        "",
        "## Artifacts",
        "",
        f"- enabled graph: {report['artifacts']['conversation_cache_enabled_ttft_svg']}",
        f"- enabled csv: {report['artifacts']['conversation_cache_enabled_ttft_csv']}",
        f"- disabled graph: {report['artifacts']['conversation_cache_disabled_ttft_svg']}",
        f"- disabled csv: {report['artifacts']['conversation_cache_disabled_ttft_csv']}",
        "",
    ]
    (out_root / "benchmark_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    env_script = args.env_script or platform_env_script(args.platform)
    if not env_script.is_file():
        raise FileNotFoundError(f"missing platform env script: {env_script}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = (
        args.out_dir
        or ROOT / "benchmarks" / f"{args.platform}-conversation-cache-{timestamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    base_env = load_platform_env(env_script)
    bot_pid_path = out_root / "shared" / "benchmark-bot.pid"
    bot_pid_path.parent.mkdir(parents=True, exist_ok=True)
    run_order: list[dict[str, Any]] = []
    success_counts = {config.name: 0 for config in CONFIGS}
    attempt_index = 0

    ensure_vllm_ready(base_env, timeout_secs=args.vllm_ready_timeout_secs)
    stop_bot(base_env, label="pre-benchmark cleanup")

    try:
        while any(count < args.pairs for count in success_counts.values()):
            attempt_index += 1
            for config in CONFIGS:
                if success_counts[config.name] >= args.pairs:
                    continue
                result = run_pair_member(
                    base_env=base_env,
                    bot_pid_path=bot_pid_path,
                    out_root=out_root,
                    pair_index=attempt_index,
                    config=config,
                    turn_pause_secs=args.turn_pause_secs,
                )
                if result["returncode"] == 0:
                    success_counts[config.name] += 1
                run_order.append(
                    {
                        "attempt_index": attempt_index,
                        "config": config.name,
                        "run_dir": str(result["run_dir"].relative_to(out_root)),
                        "returncode": result["returncode"],
                        "successful_runs_for_config": success_counts[config.name],
                    }
                )
    finally:
        cleanup_env = base_env.copy()
        cleanup_env["NEMOTRON_BOT_PID"] = str(bot_pid_path)
        stop_bot(cleanup_env, label="final cleanup")

    (out_root / "run_order.json").write_text(
        json.dumps(run_order, indent=2) + "\n",
        encoding="utf-8",
    )
    report = analyze_runs(out_root, pairs=args.pairs)
    write_report(out_root, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
