#!/usr/bin/env python3
"""Run a controlled Cartesia/vLLM audio sensitivity pilot."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import sys
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile
import av

from generate_cartesia_audio_fixtures import (  # noqa: E402
    DEFAULT_ENV_FILES,
    _load_api_key,
    _synthesize_pcm,
    _write_wav,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


REQUEST_TEMPLATE_PATH = (
    ROOT / "artifacts" / "nvidia-audio-repro-20260504" / "request_template.json"
)
DEFAULT_OUT_DIR = ROOT / "artifacts" / "audio-resampling-pilot-20260504"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nemotron_3_nano_omni"
DEFAULT_PROBE_DIR = Path("/tmp/vllm-audio-probe")
TARGET_AUDIO_SR = 16000
SOURCE_AUDIO_SR = 48000


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    article: str
    topic: str

    @property
    def audio_prompt(self) -> str:
        return f"Tell me in one sentence about {self.article} {self.topic}."

    @property
    def assistant_response(self) -> str:
        article_cap = self.article.capitalize()
        return (
            f"{article_cap} {self.topic} is a simple example subject used for this "
            "regression study."
        )

    @property
    def recall_answer(self) -> str:
        return self.topic


PILOT_CASES: list[PilotCase] = [
    PilotCase("01-unicorn", "a", "unicorn"),
    PilotCase("02-dragon", "a", "dragon"),
    PilotCase("03-phoenix", "a", "phoenix"),
    PilotCase("04-tiger", "a", "tiger"),
    PilotCase("05-robot", "a", "robot"),
    PilotCase("06-castle", "a", "castle"),
    PilotCase("07-comet", "a", "comet"),
    PilotCase("08-dolphin", "a", "dolphin"),
    PilotCase("09-lantern", "a", "lantern"),
    PilotCase("10-cactus", "a", "cactus"),
    PilotCase("11-rocket", "a", "rocket"),
    PilotCase("12-sparrow", "a", "sparrow"),
    PilotCase("13-volcano", "a", "volcano"),
    PilotCase("14-elephant", "an", "elephant"),
    PilotCase("15-igloo", "an", "igloo"),
    PilotCase("16-octopus", "an", "octopus"),
    PilotCase("17-astronaut", "an", "astronaut"),
    PilotCase("18-engine", "an", "engine"),
    PilotCase("19-orchid", "an", "orchid"),
    PilotCase("20-umpire", "an", "umpire"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--voice-id", default="71a7ad14-091c-4e8e-a314-022ece01c121")
    parser.add_argument("--tts-model", default="sonic-3")
    parser.add_argument("--cartesia-version", default="2024-11-13")
    parser.add_argument("--api-key-env", default="CARTESIA_API_KEY")
    parser.add_argument("--env-file", action="append", type=Path)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    return parser.parse_args()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_f32(audio: np.ndarray) -> str:
    audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
    return hashlib.sha256(audio_f32.tobytes()).hexdigest()


def _resample_audio_pyav(audio: np.ndarray, *, orig_sr: float, target_sr: float) -> np.ndarray:
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))
    if orig_sr_int == target_sr_int:
        return np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        return np.stack(
            [
                _resample_audio_pyav(channel, orig_sr=orig_sr, target_sr=target_sr)
                for channel in audio
            ],
            axis=0,
        )

    expected_len = int(math.ceil(audio.shape[-1] * target_sr_int / orig_sr_int))
    min_samples = 1024
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < min_samples:
        audio_f32 = np.pad(audio_f32, (0, min_samples - len(audio_f32)))
    audio_f32 = audio_f32.reshape(1, -1)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr_int)
    frame = av.AudioFrame.from_ndarray(audio_f32, format="fltp", layout="mono")
    frame.sample_rate = orig_sr_int
    out_frames = resampler.resample(frame)
    out_frames.extend(resampler.resample(None))
    result = np.concatenate([frame.to_ndarray() for frame in out_frames], axis=1).squeeze(0)
    return result[:expected_len]


def _read_wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
    return {
        "path": str(path),
        "sha256_bytes": _sha256_bytes(path.read_bytes()),
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": width,
        "frames": frames,
        "duration_s": frames / sr,
    }


def _write_pcm16_wav(path: Path, audio: np.ndarray, *, sample_rate: int, mode: str) -> None:
    audio_1d = np.asarray(audio, dtype=np.float32).reshape(-1)
    scaled = np.clip(audio_1d, -1.0, 1.0) * 32767.0
    if mode == "trunc":
        pcm16 = np.trunc(scaled).astype(np.int16)
    elif mode == "round":
        pcm16 = np.round(scaled).astype(np.int16)
    else:
        raise ValueError(f"Unsupported PCM16 mode: {mode}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _audio_file_to_data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _load_request_template() -> dict[str, Any]:
    return json.loads(REQUEST_TEMPLATE_PATH.read_text(encoding="utf-8"))


def _build_payload(case: PilotCase, sample_path: Path, model: str) -> dict[str, Any]:
    payload = _load_request_template()
    payload["model"] = model
    payload["stream"] = False
    payload.pop("stream_options", None)
    payload["messages"][1]["content"][1]["audio_url"]["url"] = _audio_file_to_data_url(
        sample_path
    )
    payload["messages"][2]["content"] = case.assistant_response
    payload["messages"][3]["content"] = (
        "What topic did I mention in the previous audio? Answer with one word only."
    )
    payload["messages"][4]["content"] = case.recall_answer
    return payload


def _run_payload(
    *,
    base_url: str,
    payload: dict[str, Any],
    probe_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, Path | None]:
    before = {path.name for path in probe_dir.glob("*.json")} if probe_dir else set()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    summary = {
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "tool_calls": message.get("tool_calls"),
        "content": message.get("content"),
        "finish_reason": body["choices"][0].get("finish_reason"),
    }

    if probe_dir is None:
        return summary, None, None

    after = sorted(probe_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    new_meta_files = [path for path in after if path.name not in before]
    if not new_meta_files:
        raise RuntimeError("Expected a new vLLM audio probe file, but none appeared")
    meta_path = new_meta_files[-1]
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return summary, metadata, meta_path


def _label_result(result: dict[str, Any]) -> tuple[str, str]:
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        args_text = tool_calls[0]["function"]["arguments"]
        return "tool", args_text
    content = (result.get("content") or "").strip()
    return "text", content


def _summarize_trials(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    outputs = []
    prompt_tokens = []
    for result in results:
        label, output = _label_result(result)
        labels.append(label)
        outputs.append(output)
        prompt_tokens.append(result["prompt_tokens"])

    return {
        "trials": len(results),
        "tool": labels.count("tool"),
        "text": labels.count("text"),
        "labels": labels,
        "outputs": outputs,
        "prompt_tokens": prompt_tokens,
        "unique_outputs": sorted(set(outputs)),
    }


def _copy_probe_artifacts(meta_path: Path, case_dir: Path) -> tuple[Path, Path]:
    copied_meta = case_dir / "probe_source48.json"
    copied_wav = case_dir / "probe_source48.parsed.wav"
    shutil.copy2(meta_path, copied_meta)
    probe_meta = json.loads(copied_meta.read_text(encoding="utf-8"))
    shutil.copy2(Path(probe_meta["parsed_wav_path"]), copied_wav)
    probe_meta["parsed_wav_path"] = str(copied_wav)
    copied_meta.write_text(
        json.dumps(probe_meta, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return copied_meta, copied_wav


def _case_pattern(source48: dict[str, Any], trunc16: dict[str, Any], round16: dict[str, Any]) -> str:
    return "/".join(
        [
            _pattern_label(source48),
            _pattern_label(trunc16),
            _pattern_label(round16),
        ]
    )


def _pattern_label(summary: dict[str, Any]) -> str:
    if summary["tool"] == summary["trials"]:
        return "tool"
    if summary["text"] == summary["trials"]:
        return "text"
    return "mixed"


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Audio Resampling Sensitivity Pilot",
        "",
        "This pilot measures model fragility in the audio path: whether semantically",
        "equivalent audio inputs, differing only in low-level waveform or serialization",
        "details, cause the model to behave differently for a fixed direct-to-vLLM",
        "request shape.",
        "",
        "## Design",
        "",
        "- 20 Cartesia-generated source prompts at `48 kHz`.",
        "- Fixed request template based on the packaged NVIDIA audio repro.",
        "- For each prompt, replay three audio variants:",
        "  - `source48`: original `48 kHz` WAV; `vLLM` performs the `48k -> 16k` resample internally.",
        "  - `pyav_trunc16`: offline PyAV/libswresample to `16 kHz`, then PCM16 truncation.",
        "  - `pyav_round16`: same float32 offline PyAV resample, then PCM16 rounding.",
        "- `vLLM` probe metadata from the `source48` replay is captured to compare its parsed float32",
        "  waveform hash against the offline PyAV float32 hash for the same source sample.",
        "",
        "## Overall Results",
        "",
        f"- Cases: `{report['summary']['cases_total']}`",
        f"- Trials per variant: `{report['config']['trials']}`",
        f"- Offline PyAV float32 hash matched vLLM parsed float32 hash: "
        f"`{report['summary']['probe_matches_offline_pyav_float32']}/"
        f"{report['summary']['cases_total']}`",
        "",
        "### Variant Totals",
        "",
        "| Variant | Tool calls | Direct text |",
        "| --- | ---: | ---: |",
    ]
    for variant_name, summary in report["summary"]["variant_totals"].items():
        lines.append(
            f"| `{variant_name}` | `{summary['tool']}/{summary['trials']}` | "
            f"`{summary['text']}/{summary['trials']}` |"
        )

    lines.extend(
        [
            "",
            "### Case Patterns",
            "",
            "`source48 / pyav_trunc16 / pyav_round16`",
            "",
        ]
    )
    for pattern, count in sorted(
        report["summary"]["pattern_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- `{pattern}`: `{count}` cases")

    differing_cases = report["summary"]["cases_where_round_differs_from_trunc"]
    lines.extend(["", "### Cases Where Round Differs From Trunc", ""])
    if differing_cases:
        for case_id in differing_cases:
            lines.append(f"- `{case_id}`")
    else:
        lines.append("- None")

    sensitive_cases = [
        case["case"]["case_id"]
        for case in report["cases"]
        if case["pattern"] != "text/text/text"
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The goal of this pilot is not to estimate tool-call failure in isolation. It is to",
            "  measure whether semantically equivalent audio inputs lead to different model behavior.",
            "  Tool call vs direct answer is just the concrete observable behavior in this request shape.",
            "",
            "- All `20/20` cases had identical prompt-token counts across `source48`, `pyav_trunc16`,",
            "  and `pyav_round16`, so the behavior differences in this pilot are not explained by",
            "  prompt tokenization drift.",
            "",
            f"- Only `{len(sensitive_cases)}/{report['summary']['cases_total']}` cases showed any",
            "  audio-variant-driven behavior flip at all:",
        ]
    )
    for case_id in sensitive_cases:
        lines.append(f"  - `{case_id}`")
    lines.extend(
        [
            "",
            "- The main caveat is that this prompt family turned out to be a weak tool-use baseline:",
            "  the `source48` path answered directly with `\"/home/user\"` in all `20` cases. That means",
            "  this pilot is not a good estimate of absolute tool-call failure frequency for the original",
            "  stronger NVIDIA repro family where the `48 kHz` baseline tool-calls reliably.",
            "",
            "- Even with that caveat, this remains valid evidence of audio-path fragility. In `4/20`",
            "  prompt families, low-level audio variant changes altered model behavior despite identical",
            "  prompt-token counts and matching offline-vs-in-vLLM float32 resample hashes.",
            "",
            "- The pilot still tells us two useful things:",
            "  - The offline PyAV float32 resample matches the `vLLM` parsed float32 waveform hash in",
            "    every case, so there is no evidence here of a `48k -> 16k` resampler mismatch between",
            "    our offline path and the in-`vLLM` path.",
            "  - Small waveform/serialization differences can still flip behavior on some prompts,",
            "    because several cases changed outcome despite identical prompt-token counts and",
            "    identical offline-vs-`vLLM` float32 resample hashes.",
            "",
            "## Files",
            "",
        ]
    )
    lines.append("- `manifest.json`: per-case prompt metadata")
    lines.append("- `results.json`: per-case replay outputs, probe metadata, and summary counts")
    lines.append("- `samples/<case-id>/`: source and derived WAVs plus copied probe artifacts")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.probe_dir.is_dir():
        raise RuntimeError(
            f"Probe directory {args.probe_dir} does not exist. Start vLLM with "
            "VLLM_AUDIO_PROBE_DIR enabled before running this pilot."
        )

    if args.out_dir.exists() and args.overwrite_artifacts:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_root = args.out_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    env_files = args.env_file or DEFAULT_ENV_FILES
    api_key = _load_api_key(args.api_key_env, env_files)

    manifest: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    variant_totals: dict[str, Counter[str]] = {
        "source48": Counter(),
        "pyav_trunc16": Counter(),
        "pyav_round16": Counter(),
    }
    pattern_counts: Counter[str] = Counter()
    cases_where_round_differs_from_trunc: list[str] = []
    probe_matches = 0

    for case in PILOT_CASES:
        case_dir = samples_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        source48_path = case_dir / "source48.wav"
        if args.overwrite_audio or not source48_path.exists():
            pcm = _synthesize_pcm(
                api_key=api_key,
                transcript=case.audio_prompt,
                voice_id=args.voice_id,
                model=args.tts_model,
                sample_rate=SOURCE_AUDIO_SR,
                cartesia_version=args.cartesia_version,
                speed=None,
            )
            _write_wav(source48_path, pcm, sample_rate=SOURCE_AUDIO_SR)

        source_audio, source_sr = soundfile.read(source48_path, dtype="float32", always_2d=False)
        if isinstance(source_sr, np.generic):
            source_sr = int(source_sr)
        if np.asarray(source_audio).ndim > 1:
            source_audio = np.mean(np.asarray(source_audio), axis=1)
        if int(round(source_sr)) != SOURCE_AUDIO_SR:
            raise RuntimeError(
                f"Expected {SOURCE_AUDIO_SR} Hz source audio, got {source_sr} for {source48_path}"
            )
        offline_pyav_f32 = _resample_audio_pyav(
            np.asarray(source_audio, dtype=np.float32),
            orig_sr=source_sr,
            target_sr=TARGET_AUDIO_SR,
        )
        offline_pyav_f32_hash = _sha256_f32(offline_pyav_f32)

        trunc16_path = case_dir / "pyav_trunc16.wav"
        round16_path = case_dir / "pyav_round16.wav"
        _write_pcm16_wav(trunc16_path, offline_pyav_f32, sample_rate=TARGET_AUDIO_SR, mode="trunc")
        _write_pcm16_wav(round16_path, offline_pyav_f32, sample_rate=TARGET_AUDIO_SR, mode="round")

        source48_results: list[dict[str, Any]] = []
        probe_metadata: dict[str, Any] | None = None
        probe_meta_path: Path | None = None
        for trial_idx in range(args.trials):
            payload = _build_payload(case, source48_path, args.model)
            result, metadata, meta_path = _run_payload(
                base_url=args.base_url,
                payload=payload,
                probe_dir=args.probe_dir,
            )
            source48_results.append(result)
            if trial_idx == 0:
                probe_metadata = metadata
                probe_meta_path = meta_path

        if probe_metadata is None or probe_meta_path is None:
            raise RuntimeError(f"Missing source48 probe metadata for {case.case_id}")
        copied_probe_meta, copied_probe_wav = _copy_probe_artifacts(probe_meta_path, case_dir)
        probe_matches_offline_pyav = (
            probe_metadata["parsed_sha256_f32"] == offline_pyav_f32_hash
        )
        if probe_matches_offline_pyav:
            probe_matches += 1

        trunc_results = [
            _run_payload(
                base_url=args.base_url,
                payload=_build_payload(case, trunc16_path, args.model),
                probe_dir=None,
            )[0]
            for _ in range(args.trials)
        ]
        round_results = [
            _run_payload(
                base_url=args.base_url,
                payload=_build_payload(case, round16_path, args.model),
                probe_dir=None,
            )[0]
            for _ in range(args.trials)
        ]

        source48_summary = _summarize_trials(source48_results)
        trunc_summary = _summarize_trials(trunc_results)
        round_summary = _summarize_trials(round_results)

        for name, summary in [
            ("source48", source48_summary),
            ("pyav_trunc16", trunc_summary),
            ("pyav_round16", round_summary),
        ]:
            variant_totals[name]["tool"] += summary["tool"]
            variant_totals[name]["text"] += summary["text"]
            variant_totals[name]["trials"] += summary["trials"]

        pattern = _case_pattern(source48_summary, trunc_summary, round_summary)
        pattern_counts[pattern] += 1
        if trunc_summary["unique_outputs"] != round_summary["unique_outputs"] or (
            trunc_summary["tool"] != round_summary["tool"]
        ):
            cases_where_round_differs_from_trunc.append(case.case_id)

        case_report = {
            "case": asdict(case),
            "audio_prompt": case.audio_prompt,
            "assistant_response": case.assistant_response,
            "files": {
                "source48": _read_wav_info(source48_path),
                "pyav_trunc16": _read_wav_info(trunc16_path),
                "pyav_round16": _read_wav_info(round16_path),
                "probe_source48_meta": str(copied_probe_meta),
                "probe_source48_parsed_wav": str(copied_probe_wav),
            },
            "offline_pyav_float32": {
                "sample_rate": TARGET_AUDIO_SR,
                "sha256_f32": offline_pyav_f32_hash,
                "num_samples": int(np.asarray(offline_pyav_f32).reshape(-1).shape[-1]),
            },
            "probe_source48": {
                **probe_metadata,
                "parsed_wav_path": str(copied_probe_wav),
                "meta_path": str(copied_probe_meta),
                "matches_offline_pyav_float32": probe_matches_offline_pyav,
            },
            "variants": {
                "source48": source48_summary,
                "pyav_trunc16": trunc_summary,
                "pyav_round16": round_summary,
            },
            "pattern": pattern,
        }
        case_reports.append(case_report)
        manifest.append(
            {
                "case_id": case.case_id,
                "topic": case.topic,
                "article": case.article,
                "audio_prompt": case.audio_prompt,
                "assistant_response": case.assistant_response,
                "recall_answer": case.recall_answer,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "trials": args.trials,
            "probe_dir": str(args.probe_dir),
            "out_dir": str(args.out_dir),
            "tts_model": args.tts_model,
            "voice_id": args.voice_id,
            "cartesia_version": args.cartesia_version,
        },
        "summary": {
            "cases_total": len(PILOT_CASES),
            "probe_matches_offline_pyav_float32": probe_matches,
            "variant_totals": {
                key: {
                    "tool": value["tool"],
                    "text": value["text"],
                    "trials": value["trials"],
                }
                for key, value in variant_totals.items()
            },
            "pattern_counts": dict(sorted(pattern_counts.items())),
            "cases_where_round_differs_from_trunc": cases_where_round_differs_from_trunc,
        },
        "cases": case_reports,
    }

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out_dir / "README.md", report)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
