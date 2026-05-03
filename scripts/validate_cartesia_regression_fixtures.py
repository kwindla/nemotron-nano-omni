#!/usr/bin/env python3
"""Replay Cartesia regression fixtures and validate Smart Turn behavior."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOT_LOG = ROOT / "logs" / "bot.log"
AIORTC_CLIENT = ROOT / ".venv-pipecat" / "bin" / "python"
AIORTC_CLIENT_SCRIPT = ROOT / "scripts" / "nemotron_omni_aiortc_client.py"
FIXTURE_DIR = ROOT / "media" / "cartesia-regression"


@dataclass(frozen=True)
class FixtureValidationSpec:
    filename: str
    expected_response_substrings: tuple[str, ...]
    require_tool: bool = False


@dataclass
class FixtureRunResult:
    filename: str
    repetition: int
    user_turns: int
    cancel_count: int
    final_transcripts: list[str]
    completed_responses: list[str]
    tool_commands: list[str]


DEFAULT_SPECS = [
    FixtureValidationSpec(
        filename="audio_unicorn_intro.wav",
        expected_response_substrings=("unicorn",),
    ),
    FixtureValidationSpec(
        filename="audio_dragon_intro.wav",
        expected_response_substrings=("dragon",),
    ),
    FixtureValidationSpec(
        filename="audio_tool_echo_one.wav",
        expected_response_substrings=(
            "spark audio 1|spark audio one|spark_audio_1|audio 1|audio one",
        ),
        require_tool=True,
    ),
    FixtureValidationSpec(
        filename="audio_tool_echo_three.wav",
        expected_response_substrings=(
            "spark audio 3|spark audio three|spark_audio_3|audio 3|audio three",
        ),
        require_tool=True,
    ),
    FixtureValidationSpec(
        filename="audio_math_1000_div_25.wav",
        expected_response_substrings=("40",),
    ),
    FixtureValidationSpec(
        filename="audio_goodbye.wav",
        expected_response_substrings=("goodbye",),
    ),
    FixtureValidationSpec(
        filename="audio_tool_echo_five.wav",
        expected_response_substrings=(
            "finalaudiofive|final audio5|audio5|final audio 5|final audio five|final_audio_5|final_audio_five",
        ),
        require_tool=True,
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_unique_final_transcripts(log_text: str) -> list[str]:
    transcripts: list[str] = []
    for line in log_text.splitlines():
        if "TranscriptionFrame" not in line or "finalized: True" not in line:
            continue
        match = re.search(r"text: '([^']+)'", line)
        if not match:
            continue
        value = match.group(1)
        if value not in transcripts:
            transcripts.append(value)
    return transcripts


def _extract_completed_responses(log_text: str) -> list[str]:
    responses: list[str] = []
    for line in log_text.splitlines():
        if "completed response in" not in line:
            continue
        match = re.search(r"completed response in [^:]+:\s*(.+)$", line)
        if not match:
            continue
        literal = match.group(1).strip()
        if not literal:
            continue
        try:
            value = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            value = literal
        responses.append(str(value))
    return responses


def _extract_tool_commands(log_text: str) -> list[str]:
    commands: list[str] = []
    for line in log_text.splitlines():
        if "running bash tool in" not in line:
            continue
        match = re.search(r": '([^']+)'$", line)
        if match:
            commands.append(match.group(1))
    return commands


def _contains_expected_substring(values: list[str], expected_substrings: tuple[str, ...]) -> bool:
    normalized_values = [_normalize(value) for value in values]
    for expectation in expected_substrings:
        alternatives = [_normalize(item) for item in expectation.split("|")]
        if any(
            alternative and alternative in normalized_value
            for normalized_value in normalized_values
            for alternative in alternatives
        ):
            return True
    return False


def _validate_run(spec: FixtureValidationSpec, result: FixtureRunResult) -> None:
    if result.user_turns != 1:
        raise AssertionError(
            f"{spec.filename} repetition {result.repetition}: expected exactly 1 user turn, "
            f"found {result.user_turns}"
        )
    if result.cancel_count != 0:
        raise AssertionError(
            f"{spec.filename} repetition {result.repetition}: expected no cancelled audio "
            f"completions, found {result.cancel_count}"
        )
    if len(result.final_transcripts) != 1:
        raise AssertionError(
            f"{spec.filename} repetition {result.repetition}: expected exactly 1 final "
            f"transcript, found {len(result.final_transcripts)} ({result.final_transcripts})"
        )
    if spec.require_tool and len(result.tool_commands) != 1:
        raise AssertionError(
            f"{spec.filename} repetition {result.repetition}: expected exactly 1 tool command, "
            f"found {len(result.tool_commands)} ({result.tool_commands})"
        )
    if not _contains_expected_substring(
        result.completed_responses, spec.expected_response_substrings
    ):
        raise AssertionError(
            f"{spec.filename} repetition {result.repetition}: response mismatch. "
            f"Expected one of {spec.expected_response_substrings}, got "
            f"{result.completed_responses}"
        )


def _run_fixture_once(
    *,
    spec: FixtureValidationSpec,
    repetition: int,
    run_secs: float,
    silence_secs: float,
) -> FixtureRunResult:
    if not BOT_LOG.exists():
        raise RuntimeError(f"Bot log not found: {BOT_LOG}")
    bot_log_offset = BOT_LOG.stat().st_size
    fixture_path = FIXTURE_DIR / spec.filename
    if not fixture_path.exists():
        raise RuntimeError(f"Fixture not found: {fixture_path}")

    subprocess.run(
        [
            str(AIORTC_CLIENT),
            str(AIORTC_CLIENT_SCRIPT),
            "--audio",
            str(fixture_path),
            "--run-secs",
            str(run_secs),
            "--silence-secs",
            str(silence_secs),
            "--turn-gap-secs",
            "8",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    log_text = BOT_LOG.read_text(encoding="utf-8", errors="replace")[bot_log_offset:]
    return FixtureRunResult(
        filename=spec.filename,
        repetition=repetition,
        user_turns=len(re.findall(r"Added user audio turn to LLM context", log_text)),
        cancel_count=len(re.findall(r"audio completion cancelled", log_text)),
        final_transcripts=_extract_unique_final_transcripts(log_text),
        completed_responses=_extract_completed_responses(log_text),
        tool_commands=_extract_tool_commands(log_text),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="How many times to replay each audio fixture.",
    )
    parser.add_argument(
        "--run-secs",
        type=float,
        default=18.0,
        help="How long the aiortc client should stay connected per replay.",
    )
    parser.add_argument(
        "--silence-secs",
        type=float,
        default=6.0,
        help="How long the aiortc client should send trailing silence.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        help="Optional fixture filename to validate. May be repeated.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path for JSON results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = DEFAULT_SPECS
    if args.fixture:
        selected = set(args.fixture)
        specs = [spec for spec in specs if spec.filename in selected]
        missing = selected.difference(spec.filename for spec in specs)
        if missing:
            raise RuntimeError(f"Unknown fixture(s): {sorted(missing)}")

    results: list[FixtureRunResult] = []
    for spec in specs:
        print(f"validate {spec.filename}")
        for repetition in range(1, args.repetitions + 1):
            result = _run_fixture_once(
                spec=spec,
                repetition=repetition,
                run_secs=args.run_secs,
                silence_secs=args.silence_secs,
            )
            _validate_run(spec, result)
            results.append(result)
            print(
                f"  repetition {repetition}: user_turns={result.user_turns} "
                f"final_transcript={result.final_transcripts[0]!r}"
            )

    if args.json:
        payload = {
            "repetitions": args.repetitions,
            "results": [asdict(result) for result in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
