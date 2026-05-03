"""Tests for the local Pocket TTS WAV streaming parser."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import pytest

from nemotron_voice.services.kyutai.tts import PocketTTSService
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame


async def _iter_chunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


class _FakeContent:
    def __init__(self, chunks: list[bytes], *, exc: Exception | None = None):
        self._chunks = chunks
        self._exc = exc

    async def iter_chunked(self, _: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._exc is not None:
            raise self._exc


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        chunks: list[bytes] | None = None,
        exc: Exception | None = None,
        text_body: str = "",
    ):
        self.status = status
        self.content = _FakeContent(chunks or [], exc=exc)
        self._text_body = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self) -> str:
        return self._text_body


class _FakeSession:
    def __init__(self, outcomes: list[object]):
        self._outcomes = list(outcomes)
        self.closed = False
        self.post_calls = 0
        self.close_calls = 0

    def post(self, *_, **__):
        self.post_calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self):
        self.closed = True
        self.close_calls += 1


class _SessionFactory:
    def __init__(self, sessions: list[_FakeSession]):
        self._sessions = list(sessions)
        self.created = 0

    def __call__(self):
        session = self._sessions[self.created]
        self.created += 1
        return session


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


@pytest.mark.asyncio
async def test_pocket_tts_retries_stale_disconnect_before_audio(monkeypatch):
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    header = _wav_header(sample_rate=24000, pcm_size=len(pcm))
    chunks = [header[:12], header[12:], pcm]

    first_session = _FakeSession([aiohttp.ServerDisconnectedError()])
    second_session = _FakeSession([_FakeResponse(chunks=chunks)])
    factory = _SessionFactory([first_session, second_session])
    monkeypatch.setattr("nemotron_voice.services.kyutai.tts.aiohttp.ClientSession", factory)

    service = PocketTTSService(sample_rate=24000)
    service._sample_rate = 24000

    frames = [frame async for frame in service.run_tts("retry me", "ctx")]

    assert factory.created == 2
    assert first_session.close_calls == 1
    assert second_session.post_calls == 1
    assert not any(isinstance(frame, ErrorFrame) for frame in frames)
    audio = b"".join(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))
    assert audio == pcm


@pytest.mark.asyncio
async def test_pocket_tts_does_not_retry_disconnect_after_audio(monkeypatch):
    pcm = b"\x10\x00\x20\x00\x30\x00\x40\x00"
    header = _wav_header(sample_rate=24000, pcm_size=len(pcm))
    chunks = [header[:20], header[20:], pcm]

    session = _FakeSession(
        [
            _FakeResponse(
                chunks=chunks,
                exc=aiohttp.ServerDisconnectedError(),
            )
        ]
    )
    factory = _SessionFactory([session])
    monkeypatch.setattr("nemotron_voice.services.kyutai.tts.aiohttp.ClientSession", factory)

    service = PocketTTSService(sample_rate=24000)
    service._sample_rate = 24000

    frames = [frame async for frame in service.run_tts("partial", "ctx")]

    assert factory.created == 1
    assert session.post_calls == 1
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)
    errors = [frame for frame in frames if isinstance(frame, ErrorFrame)]
    assert len(errors) == 1
    assert errors[0].error == "local Pocket TTS error: Server disconnected"
