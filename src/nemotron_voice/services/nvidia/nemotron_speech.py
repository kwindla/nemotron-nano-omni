#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local Nemotron Speech WebSocket STT service.

This service speaks the lightweight protocol used by the local
``nemotron_speech.server`` implementation from the January Nemotron voice-agent
repo:

* binary messages are 16-bit PCM, 16 kHz, mono audio chunks
* ``{"type": "reset", "finalize": true}`` finalizes the current utterance
* transcript messages have ``type``, ``text``, and ``is_final`` fields

It is intentionally separate from :mod:`pipecat.services.nvidia.stt`, which is a
Riva/NIM gRPC client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import DEFAULT_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use local Nemotron Speech STT, install the websockets package.")
    raise Exception(f"Missing module: {e}")


class NemotronSpeechWebSocketSTTService(WebsocketSTTService):
    """STT client for the local Nemotron Speech WebSocket ASR server."""

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:8080",
        sample_rate: int | None = 16000,
        language: Language | None = Language.EN_US,
        audio_passthrough: bool = False,
        finalize_on_vad: bool = False,
        finalize_on_user_stop: bool = True,
        ready_timeout_secs: float = 10.0,
        ttfs_p99_latency: float | None = DEFAULT_TTFS_P99,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            url: WebSocket URL for the local ASR server.
            sample_rate: Incoming PCM sample rate. The local ASR server expects 16 kHz.
            language: Language attached to transcription frames.
            audio_passthrough: Whether audio frames should continue downstream after STT.
            finalize_on_vad: Whether VAD stop should send a hard reset/finalize command.
            finalize_on_user_stop: Whether completed user turns should finalize ASR.
            ready_timeout_secs: Seconds to wait for the server ready message.
            ttfs_p99_latency: Metadata for downstream turn processors.
            **kwargs: Additional arguments passed to ``WebsocketSTTService``.
        """
        settings = kwargs.pop("settings", STTSettings(model=None, language=language))
        super().__init__(
            sample_rate=sample_rate,
            audio_passthrough=audio_passthrough,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=settings,
            **kwargs,
        )
        self._url = url
        self._language = language
        self._finalize_on_vad = finalize_on_vad
        self._finalize_on_user_stop = finalize_on_user_stop
        self._ready_timeout_secs = ready_timeout_secs
        self._receive_task: asyncio.Task | None = None
        self._ready = False
        self._audio_send_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self.sample_rate != 16000:
            logger.warning(
                f"{self} local Nemotron Speech ASR expects 16 kHz PCM; "
                f"received sample_rate={self.sample_rate}"
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._send_reset(finalize=True)
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send raw PCM audio to the local ASR server."""
        if self._websocket and self._websocket.state is State.OPEN and self._ready:
            try:
                async with self._audio_send_lock:
                    await self._websocket.send(audio)
            except Exception as e:
                await self._report_error(ErrorFrame(f"{self} failed to send audio: {e}"))
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._finalize_on_vad:
            await self._send_reset(finalize=True)
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._finalize_on_user_stop:
            await self._send_reset(finalize=True)

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        await self._call_event_handler("on_connected", self)

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()
        await self._call_event_handler("on_disconnected", self)

    async def _connect_websocket(self):
        if self._websocket and self._websocket.state is State.OPEN:
            return

        logger.info(f"{self} connecting to local Nemotron Speech ASR at {self._url}")
        self._websocket = await websocket_connect(self._url)
        self._ready = False

        try:
            message = await asyncio.wait_for(self._websocket.recv(), self._ready_timeout_secs)
            data = json.loads(message)
            if data.get("type") == "ready":
                self._ready = True
                logger.info(f"{self} connected to local Nemotron Speech ASR")
                return
            logger.warning(f"{self} unexpected initial ASR message: {data}")
        except asyncio.TimeoutError:
            logger.warning(f"{self} timed out waiting for ASR ready message; continuing")
        except Exception as e:
            await self._disconnect_websocket()
            raise RuntimeError(f"failed to connect to local Nemotron Speech ASR: {e}") from e

        self._ready = True

    async def _disconnect_websocket(self):
        self._ready = False
        if self._websocket:
            try:
                await self._websocket.close()
            finally:
                self._websocket = None

    async def _receive_messages(self):
        if not self._websocket:
            return

        async for message in self._websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON ASR message: {message!r}")
                continue

            msg_type = data.get("type")
            if msg_type == "ready":
                self._ready = True
                continue
            if msg_type == "transcript":
                await self._handle_transcript(data)
                continue
            if msg_type == "error":
                await self._report_error(
                    ErrorFrame(f"{self} ASR server error: {data.get('message', 'unknown error')}")
                )
                continue
            logger.debug(f"{self} ignored ASR message: {data}")

    async def _send_reset(self, *, finalize: bool):
        if not (self._websocket and self._websocket.state is State.OPEN and self._ready):
            return
        try:
            async with self._audio_send_lock:
                await self._websocket.send(json.dumps({"type": "reset", "finalize": finalize}))
        except Exception as e:
            await self._report_error(ErrorFrame(f"{self} failed to finalize ASR turn: {e}"))

    async def _handle_transcript(self, data: dict[str, Any]):
        text = data.get("text", "")
        if not text:
            return

        timestamp = time_now_iso8601()
        if data.get("is_final", False):
            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=timestamp,
                    language=self._language,
                    result=data,
                    finalized=True,
                )
            )
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=timestamp,
                    language=self._language,
                    result=data,
                )
            )
