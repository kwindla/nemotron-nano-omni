#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local NVIDIA Magpie WebSocket TTS service.

This service speaks the WebSocket protocol exposed by the local
``nemotron_speech.tts_server`` implementation from the January Nemotron
voice-agent repo:

* client sends JSON text segments to ``/ws/tts/stream``
* server sends raw 16-bit PCM, 22 kHz, mono audio in binary messages
* server sends JSON control messages for segment completion and stream end

It is intentionally separate from :mod:`pipecat.services.nvidia.tts`, which is
a Riva/NIM gRPC client.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import AsyncGenerator
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use local Nemotron Magpie TTS, install the websockets package.")
    raise Exception(f"Missing module: {e}")


MAGPIE_SAMPLE_RATE = 22000
_SENTENCE_BOUNDARY_PATTERN = re.compile(r'([.!?]["\')]*\s)')
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def _sanitize_text_for_tts(text: str) -> str:
    text = _EMOJI_PATTERN.sub("", text)
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201C", '"')
    text = text.replace("\u201D", '"')
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    return text


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []

    parts = _SENTENCE_BOUNDARY_PATTERN.split(text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and _SENTENCE_BOUNDARY_PATTERN.match(parts[i + 1]):
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        elif parts[i]:
            sentences.append(parts[i])
            i += 1
        else:
            i += 1
    return sentences or [text]


class NemotronMagpieWebSocketTTSService(WebsocketTTSService):
    """TTS client for the local NVIDIA Magpie WebSocket TTS server."""

    Settings = TTSSettings

    class InputParams(BaseModel):
        """Local Magpie TTS behavior knobs."""

        streaming_preset: str = "conservative"
        use_adaptive_mode: bool = True
        sentence_pause_ms: int = 250

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8001",
        voice: str = "aria",
        language: str | Language = "en",
        sample_rate: int | None = MAGPIE_SAMPLE_RATE,
        params: Optional[InputParams] = None,
        settings: Optional[TTSSettings] = None,
        text_aggregation_mode: TextAggregationMode | None = TextAggregationMode.SENTENCE,
        **kwargs,
    ):
        default_settings = TTSSettings(
            model="nvidia/magpie_tts_multilingual_357m",
            voice=voice,
            language=language,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_start_frame=True,
            push_text_frames=True,
            pause_frame_processing=False,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        if server_url.startswith("http://"):
            server_url = "ws://" + server_url[7:]
        elif server_url.startswith("https://"):
            server_url = "wss://" + server_url[8:]

        self._server_url = server_url.rstrip("/")
        self._params = params or self.InputParams()
        self._receive_task = None
        self._active_context_id: str | None = None
        self._stream_active = False
        self._stream_closed = False
        self._first_segment_pending = True
        self._segment_sentence_boundary_queue: deque[bool] = deque()

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        value = language.value.lower()
        if value.startswith("en"):
            return "en"
        if value.startswith("es"):
            return "es"
        if value.startswith("de"):
            return "de"
        if value.startswith("fr"):
            return "fr"
        if value.startswith("vi"):
            return "vi"
        if value.startswith("it"):
            return "it"
        if value.startswith("zh"):
            return "zh"
        return None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_url = f"{self._server_url}/ws/tts/stream"
            logger.info(f"{self} connecting to local Magpie TTS at {ws_url}")
            self._websocket = await websocket_connect(ws_url)
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "init",
                        "voice": self._settings.voice,
                        "language": self._settings.language,
                    }
                )
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")
            await self.push_error_frame(ErrorFrame(f"{self} WebSocket connection failed: {e}"))

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self} disconnecting from local Magpie TTS")
                await self._websocket.close()
        finally:
            if self._active_context_id and self.audio_context_available(self._active_context_id):
                await self.remove_audio_context(self._active_context_id)
            self._websocket = None
            self._active_context_id = None
            self._stream_active = False
            self._stream_closed = False
            self._first_segment_pending = True
            self._segment_sentence_boundary_queue.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                await self._handle_audio(message)
            elif isinstance(message, str):
                await self._handle_control_message(message)

    async def _handle_audio(self, audio: bytes):
        context_id = self._active_context_id
        if not context_id or not self.audio_context_available(context_id):
            return
        await self.append_to_audio_context(
            context_id,
            TTSAudioRawFrame(
                audio=audio,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            ),
        )

    async def _handle_control_message(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"{self} received invalid TTS control message: {message[:100]}")
            return

        msg_type = data.get("type")
        context_id = self._active_context_id

        if msg_type == "stream_created":
            logger.debug(f"{self} local Magpie TTS stream created: {data.get('stream_id')}")
        elif msg_type == "segment_complete":
            logger.debug(
                f"{self} local Magpie TTS segment complete: "
                f"{data.get('audio_ms', 0):.0f}ms audio"
            )
            if (
                context_id
                and self.audio_context_available(context_id)
                and self._segment_sentence_boundary_queue
                and self._segment_sentence_boundary_queue.popleft()
                and self._params.sentence_pause_ms > 0
            ):
                num_samples = int(self.sample_rate * self._params.sentence_pause_ms / 1000)
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(
                        audio=bytes(num_samples * 2),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    ),
                )
        elif msg_type == "done":
            logger.info(
                f"{self} local Magpie TTS stream complete: "
                f"{data.get('total_audio_ms', 0):.0f}ms audio, "
                f"{data.get('segments_generated', 0)} segments"
            )
            if context_id and self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            self._active_context_id = None
            self._stream_active = False
            self._stream_closed = False
            self._first_segment_pending = True
            self._segment_sentence_boundary_queue.clear()
        elif msg_type == "error":
            error = data.get("message", "unknown local Magpie TTS error")
            is_fatal = data.get("fatal", False)
            logger.error(f"{self} local Magpie TTS error: {error} (fatal={is_fatal})")
            if context_id and self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, ErrorFrame(error=error))
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            else:
                await self.push_error_frame(ErrorFrame(error=error))
            if is_fatal:
                self._stream_active = False
        elif msg_type == "pong":
            pass
        else:
            logger.debug(f"{self} ignored local Magpie TTS message: {data}")

    async def flush_audio(self, context_id: str | None = None):
        flush_id = context_id or self._active_context_id
        if not flush_id or flush_id != self._active_context_id:
            return
        if not self._websocket or not self._stream_active or self._stream_closed:
            return

        try:
            await self._websocket.send(json.dumps({"type": "close"}))
            self._stream_closed = True
        except Exception as e:
            logger.debug(f"{self} failed to close local Magpie TTS stream: {e}")

    async def on_audio_context_interrupted(self, context_id: str):
        await self.stop_all_metrics()
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception as e:
                logger.debug(f"{self} failed to cancel local Magpie TTS stream: {e}")
        self._active_context_id = None
        self._stream_active = False
        self._stream_closed = False
        self._first_segment_pending = True
        self._segment_sentence_boundary_queue.clear()
        await super().on_audio_context_interrupted(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = _sanitize_text_for_tts(text)
        if not text or not text.strip():
            yield None
            return

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._stream_active:
                self._active_context_id = context_id
                self._stream_active = True
                self._stream_closed = False
                self._first_segment_pending = True
            elif self._active_context_id != context_id:
                await self.flush_audio(self._active_context_id)
                self._active_context_id = context_id
                self._stream_active = True
                self._stream_closed = False
                self._first_segment_pending = True

            for segment in _split_into_sentences(text):
                if not segment or not segment.strip():
                    continue

                msg = {"type": "text", "text": segment}
                if self._params.use_adaptive_mode and self._first_segment_pending:
                    msg["mode"] = "stream"
                    msg["preset"] = self._params.streaming_preset
                    self._first_segment_pending = False
                else:
                    msg["mode"] = "batch"

                await self._get_websocket().send(json.dumps(msg))
                self._segment_sentence_boundary_queue.append(segment.strip()[-1:] in ".!?")

            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self} local Magpie TTS error: {e}")
            yield ErrorFrame(error=f"local Magpie TTS error: {e}")
