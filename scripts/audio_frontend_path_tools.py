#!/usr/bin/env python3
"""Helpers for reproducing the Pipecat SmallWebRTC and vLLM audio paths."""

from __future__ import annotations

import fractions
import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile

TARGET_SR = 16000
SOURCE_SR = 48000
PIPECAT_FRAME_MS = 20


@dataclass(frozen=True)
class GeneratedPathWavs:
    source_wav: Path
    pipecat_wav: Path
    vllm_wav: Path
    metadata_json: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_f32(audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _read_wav_info(path: Path) -> dict[str, Any]:
    info = soundfile.info(path)
    frames = int(info.frames)
    sr = int(info.samplerate)
    channels = int(info.channels)
    subtype = info.subtype
    if subtype == "PCM_16":
        width = 2
    elif subtype == "FLOAT":
        width = 4
    else:
        width = None
    return {
        "path": str(path),
        "sha256_bytes": _sha256_bytes(path.read_bytes()),
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": width,
        "subtype": subtype,
        "frames": frames,
        "duration_s": frames / sr,
    }


def _load_audio_f32(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = soundfile.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return arr.reshape(-1), int(sr)


def _load_audio_int16(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = soundfile.read(path, dtype="int16", always_2d=False)
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = np.round(np.mean(arr.astype(np.float32), axis=1)).astype(np.int16)
    return arr.reshape(-1).astype(np.int16), int(sr)


def resample_audio_vllm_pyav(
    audio: np.ndarray,
    *,
    orig_sr: int,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Mirror vLLM's whole-buffer PyAV/libswresample float32 path."""
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))

    if orig_sr_int == target_sr_int:
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.ndim == 2:
        return np.stack(
            [
                resample_audio_vllm_pyav(channel, orig_sr=orig_sr_int, target_sr=target_sr_int)
                for channel in audio
            ],
            axis=0,
        )

    expected_len = int(math.ceil(audio.shape[-1] * target_sr_int / orig_sr_int))

    min_samples = 1024
    audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio_f32) < min_samples:
        audio_f32 = np.pad(audio_f32, (0, min_samples - len(audio_f32)))
    audio_f32 = audio_f32.reshape(1, -1)

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr_int)
    frame = av.AudioFrame.from_ndarray(audio_f32, format="fltp", layout="mono")
    frame.sample_rate = orig_sr_int
    out_frames = resampler.resample(frame)
    out_frames.extend(resampler.resample(None))
    result = np.concatenate([out.to_ndarray() for out in out_frames], axis=1).squeeze(0)
    return np.asarray(result[:expected_len], dtype=np.float32)


def resample_audio_pipecat_smallwebrtc(
    pcm16_audio: np.ndarray,
    *,
    orig_sr: int,
    target_sr: int = TARGET_SR,
    frame_ms: int = PIPECAT_FRAME_MS,
) -> bytes:
    """Mirror SmallWebRTC's frame-by-frame PyAV/libswresample s16 path.

    This intentionally follows the live transport shape:
    - split the input into 20 ms frames
    - create ``AudioFrame`` objects from int16 mono chunks
    - call ``AudioResampler('s16', 'mono', 16000).resample(frame)``
    - concatenate ``processed_frame.to_ndarray().astype(np.int16).tobytes()``

    We do not flush the resampler at EOF because the live transport does not.
    """
    orig_sr_int = int(round(orig_sr))
    target_sr_int = int(round(target_sr))
    samples = np.asarray(pcm16_audio, dtype=np.int16).reshape(-1)

    if orig_sr_int == target_sr_int:
        return samples.tobytes()

    samples_per_frame = max(1, int(round(orig_sr_int * frame_ms / 1000.0)))
    resampler = av.AudioResampler("s16", "mono", target_sr_int)
    out_chunks: list[bytes] = []
    pts = 0

    for start in range(0, len(samples), samples_per_frame):
        chunk = samples[start : start + samples_per_frame]
        if len(chunk) == 0:
            continue
        frame = av.AudioFrame.from_ndarray(chunk[None, :], layout="mono")
        frame.sample_rate = orig_sr_int
        frame.pts = pts
        frame.time_base = fractions.Fraction(1, orig_sr_int)
        pts += len(chunk)

        frames_to_process = resampler.resample(frame)
        for processed_frame in frames_to_process:
            pcm_array = processed_frame.to_ndarray().astype(np.int16)
            out_chunks.append(pcm_array.tobytes())

    return b"".join(out_chunks)


def write_pipecat_smallwebrtc_wav(
    source_wav: Path,
    out_wav: Path,
    *,
    target_sr: int = TARGET_SR,
    frame_ms: int = PIPECAT_FRAME_MS,
) -> dict[str, Any]:
    pcm16_audio, orig_sr = _load_audio_int16(source_wav)
    pcm_bytes = resample_audio_pipecat_smallwebrtc(
        pcm16_audio,
        orig_sr=orig_sr,
        target_sr=target_sr,
        frame_ms=frame_ms,
    )
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm_bytes)

    info = _read_wav_info(out_wav)
    decoded = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    info["sha256_f32"] = _sha256_f32(decoded)
    info["path_kind"] = "pipecat_smallwebrtc"
    info["frame_ms"] = frame_ms
    return info


def write_vllm_pyav_wav(
    source_wav: Path,
    out_wav: Path,
    *,
    target_sr: int = TARGET_SR,
) -> dict[str, Any]:
    audio_f32, orig_sr = _load_audio_f32(source_wav)
    resampled = resample_audio_vllm_pyav(audio_f32, orig_sr=orig_sr, target_sr=target_sr)
    soundfile.write(
        out_wav,
        np.asarray(resampled, dtype=np.float32),
        target_sr,
        format="WAV",
        subtype="FLOAT",
    )
    info = _read_wav_info(out_wav)
    info["sha256_f32"] = _sha256_f32(resampled)
    info["path_kind"] = "vllm_pyav_float"
    return info


def generate_path_wavs(
    source_wav: Path,
    out_dir: Path,
    *,
    basename: str | None = None,
    target_sr: int = TARGET_SR,
    frame_ms: int = PIPECAT_FRAME_MS,
) -> GeneratedPathWavs:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = basename or source_wav.stem
    pipecat_wav = out_dir / f"{stem}.pipecat-smallwebrtc.wav"
    vllm_wav = out_dir / f"{stem}.vllm-pyav-float.wav"
    metadata_json = out_dir / f"{stem}.frontend-paths.json"

    source_info = _read_wav_info(source_wav)
    pipecat_info = write_pipecat_smallwebrtc_wav(
        source_wav,
        pipecat_wav,
        target_sr=target_sr,
        frame_ms=frame_ms,
    )
    vllm_info = write_vllm_pyav_wav(source_wav, vllm_wav, target_sr=target_sr)

    metadata = {
        "source": source_info,
        "pipecat_smallwebrtc": pipecat_info,
        "vllm_pyav_float": vllm_info,
        "notes": [
            "The Pipecat artifact matches SmallWebRTC's frame-by-frame PyAV resample to s16 mono 16 kHz, then serializes the exact int16 bytes to WAV.",
            "The vLLM artifact matches vLLM's whole-buffer PyAV resample to float32 mono 16 kHz and is written as a float WAV so no extra PCM16 quantization is introduced.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    return GeneratedPathWavs(
        source_wav=source_wav,
        pipecat_wav=pipecat_wav,
        vllm_wav=vllm_wav,
        metadata_json=metadata_json,
    )
