#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kyutai Pocket TTS service implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


POCKET_TTS_SAMPLE_RATE = 24000


@dataclass
class PocketTTSSettings(TTSSettings):
    """Settings for PocketTTSService."""

    pass


class PocketTTSService(TTSService):
    """HTTP streaming client for a local Kyutai Pocket TTS server.

    Pocket TTS exposes a FastAPI endpoint at ``POST /tts`` that streams a WAV
    response. This service posts text plus a voice selector and emits Pipecat
    raw audio frames after stripping the WAV header.
    """

    Settings = PocketTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001",
        voice: str = "alba",
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int | None = POCKET_TTS_SAMPLE_RATE,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = TextAggregationMode.SENTENCE,
        **kwargs,
    ):
        default_settings = self.Settings(
            model="kyutai/pocket-tts-without-voice-cloning",
            voice=voice,
            language=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._ensure_session()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._close_session()

    async def _close_session(self):
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
        if self._owns_session:
            self._session = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

    async def _reset_owned_session(self):
        if not self._owns_session:
            return
        await self._close_session()
        await self._ensure_session()

    def _build_request_data(self, text: str) -> aiohttp.FormData:
        data = aiohttp.FormData()
        data.add_field("text", text)
        if self._settings.voice:
            data.add_field("voice_url", str(self._settings.voice))
        return data

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        logger.debug(f"{self}: Generating Pocket TTS [{text}]")

        if not text or not text.strip():
            yield None
            return

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
        for attempt in range(2):
            audio_started = False
            await self._ensure_session()
            try:
                async with self._session.post(
                    f"{self._base_url}/tts",
                    data=self._build_request_data(text),
                    timeout=timeout,
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        yield ErrorFrame(
                            error=(
                                "Pocket TTS request failed "
                                f"(status: {response.status}, error: {error})"
                            )
                        )
                        return

                    await self.start_tts_usage_metrics(text)
                    audio_bytes = 0

                    async for frame in self._stream_wav_audio_frames_from_iterator(
                        response.content.iter_chunked(self.chunk_size),
                        context_id=context_id,
                    ):
                        if hasattr(frame, "audio"):
                            audio_bytes += len(frame.audio)
                            audio_started = True
                        await self.stop_ttfb_metrics()
                        yield frame

                    output_sample_rate = self.sample_rate or POCKET_TTS_SAMPLE_RATE
                    audio_ms = audio_bytes / (output_sample_rate * 2) * 1000
                    logger.info(f"{self} local Pocket TTS stream complete: {audio_ms:.0f}ms audio")
                    return
            except aiohttp.ServerDisconnectedError as e:
                if attempt == 0 and not audio_started and self._owns_session:
                    logger.warning(
                        f"{self} local Pocket TTS disconnected before audio; "
                        "recreating session and retrying once"
                    )
                    await self._reset_owned_session()
                    continue
                logger.error(f"{self} local Pocket TTS error: {e}")
                yield ErrorFrame(error=f"local Pocket TTS error: {e}")
                return
            except Exception as e:
                logger.error(f"{self} local Pocket TTS error: {e}")
                yield ErrorFrame(error=f"local Pocket TTS error: {e}")
                return
            finally:
                await self.stop_ttfb_metrics()

    async def _stream_wav_audio_frames_from_iterator(
        self,
        iterator,
        *,
        context_id: str,
    ) -> AsyncGenerator[Frame, None]:
        """Stream WAV chunks as raw Pipecat audio frames.

        Pocket TTS sends a WAV stream, and aiohttp can split the RIFF header
        across arbitrary chunks. This local parser keeps Pipecat itself
        read-only while still handling split headers and extra chunks such as
        LIST before the data chunk.
        """
        buffer = bytearray()
        wav_header_buffer = bytearray()
        source_sample_rate = None
        need_to_strip_wav_header = True

        async def maybe_resample(audio: bytes) -> bytes:
            if source_sample_rate and source_sample_rate != self.sample_rate:
                return await self._resampler.resample(
                    audio,
                    source_sample_rate,
                    self.sample_rate,
                )
            return audio

        def process_wav_header(chunk: bytes) -> bytes | None:
            nonlocal source_sample_rate, need_to_strip_wav_header
            wav_header_buffer.extend(chunk)

            if len(wav_header_buffer) < 12:
                return None
            if not wav_header_buffer.startswith(b"RIFF") or wav_header_buffer[8:12] != b"WAVE":
                need_to_strip_wav_header = False
                audio = bytes(wav_header_buffer)
                wav_header_buffer.clear()
                return audio

            offset = 12
            while offset + 8 <= len(wav_header_buffer):
                chunk_id = bytes(wav_header_buffer[offset : offset + 4])
                chunk_size = int.from_bytes(wav_header_buffer[offset + 4 : offset + 8], "little")
                payload_start = offset + 8
                payload_end = payload_start + chunk_size

                if chunk_id == b"fmt " and source_sample_rate is None:
                    if len(wav_header_buffer) < payload_start + 8:
                        return None
                    source_sample_rate = int.from_bytes(
                        wav_header_buffer[payload_start + 4 : payload_start + 8],
                        "little",
                    )

                if chunk_id == b"data":
                    need_to_strip_wav_header = False
                    audio = bytes(wav_header_buffer[payload_start:])
                    wav_header_buffer.clear()
                    return audio

                padded_payload_end = payload_end + (chunk_size % 2)
                if len(wav_header_buffer) < padded_payload_end:
                    return None
                offset = padded_payload_end

            return None

        async for chunk in iterator:
            if need_to_strip_wav_header:
                chunk = process_wav_header(chunk)
                if chunk is None:
                    continue

            buffer.extend(chunk)
            aligned_length = len(buffer) & ~1
            if aligned_length > 0:
                aligned_chunk = await maybe_resample(bytes(buffer[:aligned_length]))
                buffer = buffer[aligned_length:]
                if aligned_chunk:
                    yield TTSAudioRawFrame(
                        bytes(aligned_chunk),
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )

        if buffer:
            if len(buffer) % 2 == 1:
                buffer.extend(b"\x00")
            audio = await maybe_resample(bytes(buffer))
            yield TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
