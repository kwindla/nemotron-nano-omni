"""Tests for the local Pocket TTS WAV streaming parser."""

from __future__ import annotations

import pytest

from nemotron_voice.services.kyutai.tts import PocketTTSService
from pipecat.frames.frames import TTSAudioRawFrame


async def _iter_chunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def _wav_header(*, sample_rate: int, pcm_size: int) -> bytes:
    byte_rate = sample_rate * 2
    return (
        b"RIFF"
        + (36 + pcm_size).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + pcm_size.to_bytes(4, "little")
    )


@pytest.mark.asyncio
async def test_pocket_tts_strips_split_wav_header():
    service = PocketTTSService(sample_rate=24000)
    service._sample_rate = 24000

    pcm = b"\x20\x00\x71\x00\xd8\x00\xf9\x00\xa9\x00\x69\x00"
    header = _wav_header(sample_rate=24000, pcm_size=len(pcm))
    chunks = [header[:4], header[4:40], header[40:44], pcm[:5], pcm[5:]]

    frames = [
        frame
        async for frame in service._stream_wav_audio_frames_from_iterator(
            _iter_chunks(chunks),
            context_id="test",
        )
    ]

    audio = b"".join(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))
    assert audio == pcm


@pytest.mark.asyncio
async def test_pocket_tts_strips_wav_header_with_extra_chunk():
    service = PocketTTSService(sample_rate=24000)
    service._sample_rate = 24000

    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    junk = b"LIST" + (4).to_bytes(4, "little") + b"test"
    header = _wav_header(sample_rate=24000, pcm_size=len(pcm))
    header = header[:36] + junk + header[36:]
    chunks = [header[:17], header[17:43], header[43:49], header[49:], pcm]

    frames = [
        frame
        async for frame in service._stream_wav_audio_frames_from_iterator(
            _iter_chunks(chunks),
            context_id="test",
        )
    ]

    audio = b"".join(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))
    assert audio == pcm

