#!/usr/bin/env python3
"""Generate repeatable Cartesia WAV fixtures for mixed cache regressions."""

from __future__ import annotations

import argparse
import audioop
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureSpec:
    filename: str
    transcript: str
    speed: float | None = None
    volume: float | None = None
    emotion: str | None = None
    max_internal_silence_ms: int | None = None


DEFAULT_FIXTURES: list[FixtureSpec] = [
    FixtureSpec("audio_unicorn_intro.wav", "Tell me in one sentence about a unicorn."),
    FixtureSpec("audio_dragon_intro.wav", "Tell me in one sentence about a dragon."),
    FixtureSpec(
        "audio_tool_echo_one.wav",
        "Use bash tool and run echo spark audio one.",
        speed=2.0,
        max_internal_silence_ms=80,
    ),
    FixtureSpec(
        "audio_tool_echo_three.wav",
        "Use bash tool and run echo spark audio three.",
        speed=2.0,
        max_internal_silence_ms=80,
    ),
    FixtureSpec(
        "audio_math_1000_div_25.wav",
        "What is one thousand divided by twenty five?",
    ),
    FixtureSpec("audio_goodbye.wav", "Please say exactly goodbye."),
    FixtureSpec(
        "audio_tool_echo_five.wav",
        "Use bash tool and run echo finalaudiofive.",
        speed=2.0,
        max_internal_silence_ms=80,
    ),
]

DEFAULT_ENV_FILES = [
    Path.home() / "src" / "pipecat" / ".env",
    Path.home() / "src" / "nemotron-speech" / ".env",
]


def _load_api_key(env_name: str, env_files: list[Path]) -> str:
    value = os.getenv(env_name)
    if value:
        return value

    for env_file in env_files:
        if not env_file.is_file():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() != env_name:
                continue
            value = raw_value.strip().strip("'").strip('"')
            if value:
                return value

    raise RuntimeError(
        f"Missing {env_name}. Set it in the environment or provide --env-file."
    )


def _synthesize_pcm(
    *,
    api_key: str,
    transcript: str,
    voice_id: str,
    model: str,
    sample_rate: int,
    cartesia_version: str,
    speed: float | None,
    volume: float | None = None,
    emotion: str | None = None,
) -> bytes:
    payload = {
        "model_id": model,
        "transcript": transcript,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        },
        "language": "en",
    }
    generation_config = {}
    if speed is not None:
        generation_config["speed"] = speed
    if volume is not None:
        generation_config["volume"] = volume
    if emotion is not None:
        generation_config["emotion"] = emotion
    if generation_config:
        payload["generation_config"] = generation_config
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.cartesia.ai/tts/bytes",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Cartesia-Version": cartesia_version,
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cartesia request failed with {exc.code}: {detail}") from exc


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _cap_internal_silence(
    pcm: bytes,
    *,
    sample_rate: int,
    max_internal_silence_ms: int,
    sample_width: int = 2,
    frame_ms: int = 20,
    silence_rms_threshold: int = 150,
) -> bytes:
    frame_bytes = sample_rate * sample_width * frame_ms // 1000
    max_silent_frames = max(1, max_internal_silence_ms // frame_ms)

    output = bytearray()
    silence_buffer: list[bytes] = []
    speech_started = False

    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk += b"\x00" * (frame_bytes - len(chunk))
        rms = audioop.rms(chunk, sample_width)
        if rms <= silence_rms_threshold:
            silence_buffer.append(chunk)
            continue

        if not speech_started:
            speech_started = True
            silence_buffer.clear()
        elif silence_buffer:
            output.extend(b"".join(silence_buffer[:max_silent_frames]))
            silence_buffer.clear()

        output.extend(chunk)

    return bytes(output)


def _write_manifest(path: Path, fixtures: list[FixtureSpec]) -> None:
    manifest = []
    for fixture in fixtures:
        item = {"file": fixture.filename, "transcript": fixture.transcript}
        if fixture.speed is not None:
            item["speed"] = fixture.speed
        if fixture.volume is not None:
            item["volume"] = fixture.volume
        if fixture.emotion is not None:
            item["emotion"] = fixture.emotion
        if fixture.max_internal_silence_ms is not None:
            item["max_internal_silence_ms"] = fixture.max_internal_silence_ms
        manifest.append(item)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("media/cartesia-regression"),
        help="Directory where WAV fixtures will be written.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        help="Optional dotenv file to search for CARTESIA_API_KEY. May be repeated.",
    )
    parser.add_argument(
        "--api-key-env",
        default="CARTESIA_API_KEY",
        help="Environment variable name for the Cartesia API key.",
    )
    parser.add_argument(
        "--voice-id",
        default="71a7ad14-091c-4e8e-a314-022ece01c121",
        help="Cartesia voice id.",
    )
    parser.add_argument("--model", default="sonic-3")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--cartesia-version", default="2024-11-13")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_files = args.env_file or DEFAULT_ENV_FILES
    api_key = _load_api_key(args.api_key_env, env_files)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fixture in DEFAULT_FIXTURES:
        path = args.out_dir / fixture.filename
        if path.exists() and not args.overwrite:
            print(f"skip {path}")
            continue
        print(f"generate {path}")
        pcm = _synthesize_pcm(
            api_key=api_key,
            transcript=fixture.transcript,
            voice_id=args.voice_id,
            model=args.model,
            sample_rate=args.sample_rate,
            cartesia_version=args.cartesia_version,
            speed=fixture.speed,
            volume=fixture.volume,
            emotion=fixture.emotion,
        )
        if fixture.max_internal_silence_ms is not None:
            pcm = _cap_internal_silence(
                pcm,
                sample_rate=args.sample_rate,
                max_internal_silence_ms=fixture.max_internal_silence_ms,
            )
        _write_wav(path, pcm, sample_rate=args.sample_rate)

    _write_manifest(args.out_dir / "fixtures.json", DEFAULT_FIXTURES)
    print(f"wrote manifest {args.out_dir / 'fixtures.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
