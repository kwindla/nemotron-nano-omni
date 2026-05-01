#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nemotron Omni audio-input LLM service.

This service sends buffered user audio to a local OpenAI-compatible vLLM
endpoint and streams text deltas back as Pipecat LLM frames.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import copy
import io
import json
import os
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    StartFrame,
    UserAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven

DEFAULT_VOICE_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond in plain text only. Keep answers "
    "brief, direct, and conversational, usually one or two short sentences. Your "
    "replies will be read aloud by a text-to-speech system, so write natural "
    "spoken language rather than visual formatting. Do not use Markdown, bullet "
    "points, numbered lists, code blocks, tables, emojis, emoticons, decorative "
    "symbols, or special formatting. Avoid long lists. Do not mention these "
    "formatting rules unless asked. When you use a tool, treat the "
    "latest tool result as ground truth. If the tool result contains stdout and "
    "stderr sections, use both sections to answer. Some successful commands write "
    "normal help or diagnostic text to stderr, so do not say a command is missing "
    "just because useful output appears in stderr. The client separately displays "
    "raw bash commands and raw terminal output, so do not read ASCII art, borders, "
    "terminal markup, or long command output literally unless the user explicitly "
    "asks you to. Interpret the result and explain the useful meaning briefly. "
    "For cowthink or cowsay-style output, focus on the message inside the bubble "
    "and say that the command rendered it as ASCII art. The user may ask about "
    "the Unix tool cowthink; use the bash tool to inspect it when needed."
)

BASH_TOOL_NAME = "run_bash"
BASH_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": BASH_TOOL_NAME,
        "description": (
            "Execute arbitrary bash code in the local project workspace and return "
            "the command output. If only stdout or only stderr has content, the "
            "tool returns that content directly. If both streams have content, "
            "stdout is wrapped in <stdout>...</stdout> and stderr is wrapped in "
            "<stderr>...</stderr>. Some programs write normal help or diagnostic "
            "text to stderr even when they succeed. Use this when the user asks "
            "you to inspect or operate on the local machine. Examples: use "
            "`git branch --show-current` to see the current git branch; use "
            "`find . -maxdepth 1 -type f | wc -l` to count files in this directory; "
            "use `find . -type f | wc -l` to count files total in this project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Bash code to run with `bash -lc`.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ChatCompletionPassResult:
    output_text: str
    tool_calls: list[dict[str, Any]]
    first_token: bool


class ConversationCacheMissError(RuntimeError):
    pass


@dataclass
class NemotronOmniAudioLLMSettings(LLMSettings):
    """Settings for :class:`NemotronOmniAudioLLMService`.

    Parameters:
        audio_prompt: Text instruction sent alongside each captured audio turn.
        chat_template_kwargs: Extra chat-template kwargs forwarded to vLLM.
    """

    audio_prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    chat_template_kwargs: dict[str, Any] | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class NemotronOmniAudioLLMService(LLMService):
    """Audio-input LLM service for Nemotron Omni on vLLM.

    The service consumes ``LLMContextFrame`` frames, translates Pipecat's
    universal ``input_audio`` context parts into vLLM ``audio_url`` content
    parts, submits the full conversation to ``/v1/chat/completions``, and emits
    streamed ``LLMTextFrame`` output bounded by full-response frames.

    For smoke tests or simple pipelines, direct audio-frame inference can still
    be enabled with ``direct_audio_inference=True``. In normal Pipecat bots,
    prefer a context collector that calls ``LLMContext.add_audio_frames_message``
    at user-turn boundaries so every inference receives the full conversation.
    """

    Settings = NemotronOmniAudioLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        settings: Settings | None = None,
        audio_passthrough: bool = False,
        direct_audio_inference: bool = False,
        conversation_id: str | None = None,
        suffix_only_conversation: bool = True,
        pre_speech_buffer_secs: float = 0.5,
        min_audio_secs: float = 0.2,
        enable_internal_vad: bool = True,
        vad_rms_threshold: int = 300,
        vad_start_secs: float = 0.08,
        vad_stop_secs: float = 0.45,
        request_timeout_secs: float = 180.0,
        enable_bash_tool: bool = False,
        bash_tool_cwd: str | None = None,
        bash_tool_timeout_secs: float = 20.0,
        bash_tool_max_output_chars: int = 12000,
        bash_tool_max_rounds: int = 3,
        bash_tool_event_sender: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            api_key: Optional bearer token for the OpenAI-compatible endpoint.
            base_url: Endpoint base URL, usually ``http://127.0.0.1:8000/v1``.
            model: Model name exposed by vLLM.
            settings: Runtime-updatable LLM settings.
            audio_passthrough: Whether to pass input audio frames downstream.
            direct_audio_inference: Whether raw audio/VAD frames should trigger
                inference directly. Leave disabled when using LLMContextFrame.
            conversation_id: Optional stable id sent to vLLM for exact
                conversation-cache reuse across turns.
            suffix_only_conversation: After the first successful cached turn,
                send only the latest user message for the conversation id.
            pre_speech_buffer_secs: Audio retained before VAD start to avoid
                clipping the first phoneme.
            min_audio_secs: Minimum captured audio duration before submitting.
            enable_internal_vad: Whether to use RMS-based speech/silence
                detection when upstream VAD frames are not present.
            vad_rms_threshold: RMS threshold for the internal detector.
            vad_start_secs: Speech duration required to start an internal turn.
            vad_stop_secs: Silence duration required to stop an internal turn.
            request_timeout_secs: Total HTTP timeout for one streamed request.
            enable_bash_tool: Whether to expose the local ``run_bash`` tool.
            bash_tool_cwd: Working directory for bash tool calls.
            bash_tool_timeout_secs: Maximum runtime for one bash tool call.
            bash_tool_max_output_chars: Maximum stdout or stderr characters returned.
            bash_tool_max_rounds: Maximum tool-call iterations for one model response.
            bash_tool_event_sender: Optional async callback for publishing
                structured bash tool events to clients.
            **kwargs: Additional arguments for ``LLMService``.
        """
        default_settings = self.Settings(
            model="nemotron_3_nano_omni",
            system_instruction=DEFAULT_VOICE_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_tokens=256,
            top_p=None,
            top_k=1,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            audio_prompt="Listen to the audio and respond to the spoken instruction.",
            chat_template_kwargs={"enable_thinking": False},
            extra={},
        )

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._chat_completions_url = f"{self._base_url}/chat/completions"
        self._audio_passthrough = audio_passthrough
        self._direct_audio_inference = direct_audio_inference
        self._conversation_id = conversation_id
        self._suffix_only_conversation = suffix_only_conversation
        self._pre_speech_buffer_secs = pre_speech_buffer_secs
        self._min_audio_secs = min_audio_secs
        self._enable_internal_vad = enable_internal_vad
        self._vad_rms_threshold = vad_rms_threshold
        self._vad_start_secs = vad_start_secs
        self._vad_stop_secs = vad_stop_secs
        self._request_timeout_secs = request_timeout_secs
        self._enable_bash_tool = enable_bash_tool
        self._bash_tool_cwd = bash_tool_cwd or os.getcwd()
        self._bash_tool_timeout_secs = bash_tool_timeout_secs
        self._bash_tool_max_output_chars = bash_tool_max_output_chars
        self._bash_tool_max_rounds = bash_tool_max_rounds
        self._bash_tool_event_sender = bash_tool_event_sender

        self._session: aiohttp.ClientSession | None = None
        self._generation_task: asyncio.Task | None = None
        self._conversation_cache_committed = False

        self._sample_rate = 16000
        self._num_channels = 1
        self._pre_speech_buffer = bytearray()
        self._audio_buffer = bytearray()
        self._user_speaking = False
        self._utterance_submitted = False
        self._last_user_id = ""
        self._external_vad_seen = False
        self._internal_speech_secs = 0.0
        self._internal_silence_secs = 0.0

    def can_generate_metrics(self) -> bool:
        """Return whether the service emits processing, TTFB, and usage metrics."""
        return True

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run out-of-band text inference.

        The service is audio-input oriented and does not currently support
        out-of-band text-only inference.
        """
        raise NotImplementedError(f"run_inference() not supported by {self.__class__.__name__}")

    async def start(self, frame: StartFrame):
        """Start the service and initialize the HTTP client."""
        await super().start(frame)
        self._sample_rate = frame.audio_in_sample_rate
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout_secs)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self, frame: EndFrame):
        """Stop the service and close active HTTP resources."""
        await super().stop(frame)
        await self._cancel_generation_task()
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close active HTTP resources."""
        await super().cancel(frame)
        await self._cancel_generation_task()
        await self._close_session()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process audio, VAD, lifecycle, and pass-through frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context_frame(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            if self._direct_audio_inference:
                await self._handle_audio_frame(frame)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if self._direct_audio_inference:
                await self._handle_user_started_speaking(external=True)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._direct_audio_inference:
                await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._direct_audio_inference:
                await self._handle_user_turn_stopped()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._cancel_generation_task()
            self._reset_audio_buffers()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMRunFrame):
            logger.debug(f"{self}: ignoring {frame.name}; LLMContextFrame triggers inference")
        else:
            await self.push_frame(frame, direction)

    async def _handle_context_frame(self, context: LLMContext):
        messages = copy.deepcopy(context.get_messages(llm_specific_filter=self.__class__.__name__))
        if not messages:
            logger.debug(f"{self}: ignoring empty LLM context")
            return

        await self._cancel_generation_task()
        self._reset_audio_buffers()

        payload = self._build_payload_from_context_messages(messages)
        self._generation_task = self.create_task(
            self._run_completion_payload(
                payload,
                request_description=(
                    f"context with {len(payload['messages'])} messages and "
                    f"{self._count_audio_parts(payload['messages'])} audio parts"
                ),
                start_ttfb=True,
            ),
            name="nemotron_omni_context_completion",
        )

    async def _handle_audio_frame(self, frame: InputAudioRawFrame):
        if not frame.audio:
            return

        self._sample_rate = frame.sample_rate
        self._num_channels = frame.num_channels
        if isinstance(frame, UserAudioRawFrame):
            self._last_user_id = frame.user_id

        await self._process_internal_vad(frame)

        if self._user_speaking:
            self._audio_buffer.extend(frame.audio)
        else:
            self._append_pre_speech_audio(frame.audio)

    async def _handle_user_started_speaking(self, *, external: bool = False):
        if external:
            self._external_vad_seen = True
        self._internal_speech_secs = 0.0
        self._internal_silence_secs = 0.0
        if self._user_speaking:
            return

        await self._cancel_generation_task()
        self._user_speaking = True
        self._utterance_submitted = False
        self._audio_buffer = bytearray(self._pre_speech_buffer)
        self._pre_speech_buffer.clear()

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._external_vad_seen = True
        self._internal_speech_secs = 0.0
        self._internal_silence_secs = 0.0
        self._user_speaking = False
        if frame.stop_secs:
            speech_end_time = frame.timestamp - frame.stop_secs
            await self.start_ttfb_metrics(start_time=speech_end_time)
        await self._submit_current_utterance()

    async def _handle_internal_user_stopped_speaking(self):
        self._user_speaking = False
        await self.start_ttfb_metrics(start_time=time.time() - self._internal_silence_secs)
        self._internal_speech_secs = 0.0
        self._internal_silence_secs = 0.0
        await self._submit_current_utterance()

    async def _handle_user_turn_stopped(self):
        self._user_speaking = False
        await self._submit_current_utterance()

    async def _process_internal_vad(self, frame: InputAudioRawFrame):
        if not self._enable_internal_vad or self._external_vad_seen:
            return

        duration_secs = frame.num_frames / frame.sample_rate if frame.sample_rate else 0
        rms = audioop.rms(frame.audio, 2) if len(frame.audio) >= 2 else 0
        has_speech = rms >= self._vad_rms_threshold

        if self._user_speaking:
            if has_speech:
                self._internal_silence_secs = 0.0
            else:
                self._internal_silence_secs += duration_secs
                if self._internal_silence_secs >= self._vad_stop_secs:
                    logger.debug(
                        f"{self}: internal VAD stop after "
                        f"{self._internal_silence_secs:.3f}s silence"
                    )
                    await self._handle_internal_user_stopped_speaking()
            return

        if has_speech:
            self._internal_speech_secs += duration_secs
            if self._internal_speech_secs >= self._vad_start_secs:
                logger.debug(f"{self}: internal VAD start at RMS {rms}")
                await self._handle_user_started_speaking()
        else:
            self._internal_speech_secs = 0.0

    async def _submit_current_utterance(self):
        if self._utterance_submitted:
            return

        audio = bytes(self._audio_buffer)
        if len(audio) < self._min_audio_bytes:
            logger.debug(
                f"{self}: captured audio too short for inference "
                f"({len(audio)} bytes, minimum {self._min_audio_bytes})"
            )
            self._reset_audio_buffers()
            return

        self._utterance_submitted = True
        self._audio_buffer.clear()

        if self._generation_task and not self._generation_task.done():
            await self._cancel_generation_task()

        self._generation_task = self.create_task(
            self._run_audio_completion(audio, self._sample_rate, self._num_channels),
            name="nemotron_omni_audio_completion",
        )

    @property
    def _min_audio_bytes(self) -> int:
        return int(self._sample_rate * self._num_channels * 2 * self._min_audio_secs)

    def _append_pre_speech_audio(self, audio: bytes):
        self._pre_speech_buffer.extend(audio)
        max_bytes = int(self._sample_rate * self._num_channels * 2 * self._pre_speech_buffer_secs)
        if max_bytes > 0 and len(self._pre_speech_buffer) > max_bytes:
            del self._pre_speech_buffer[: len(self._pre_speech_buffer) - max_bytes]

    def _reset_audio_buffers(self):
        self._user_speaking = False
        self._utterance_submitted = False
        self._audio_buffer.clear()
        self._pre_speech_buffer.clear()
        self._external_vad_seen = False
        self._internal_speech_secs = 0.0
        self._internal_silence_secs = 0.0

    async def _cancel_generation_task(self):
        if self._generation_task:
            await self.cancel_task(self._generation_task)
            self._generation_task = None

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _wav_data_url(self, audio: bytes, sample_rate: int, num_channels: int) -> str:
        content = io.BytesIO()
        with wave.open(content, "wb") as wav:
            wav.setsampwidth(2)
            wav.setnchannels(num_channels)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)
        encoded = base64.b64encode(content.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    def _build_payload_from_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        full_messages = copy.deepcopy(messages)
        messages, requires_cache = self._conversation_payload_messages(messages)
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "_conversation_full_messages": full_messages,
        }
        if requires_cache:
            payload["conversation_require_cache"] = True
        if self._enable_bash_tool:
            payload["tools"] = [BASH_TOOL_DEFINITION]
            payload["tool_choice"] = "auto"

        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens
        if self._settings.temperature is not None:
            payload["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            payload["top_p"] = self._settings.top_p
        if self._settings.top_k is not None:
            payload["top_k"] = self._settings.top_k
        if self._settings.frequency_penalty is not None:
            payload["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty is not None:
            payload["presence_penalty"] = self._settings.presence_penalty
        if self._settings.seed is not None:
            payload["seed"] = self._settings.seed
        if self._settings.chat_template_kwargs:
            payload["chat_template_kwargs"] = self._settings.chat_template_kwargs

        if self._settings.extra:
            payload.update(self._settings.extra)

        if self._conversation_id:
            payload["conversation_id"] = self._conversation_id

        return payload

    def _conversation_payload_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if (
            not self._conversation_id
            or not self._suffix_only_conversation
            or not self._conversation_cache_committed
        ):
            return messages, False

        latest_user = self._latest_user_message(messages, require_audio=True)
        if latest_user is None:
            latest_user = self._latest_user_message(messages, require_audio=False)
        if latest_user is None:
            logger.warning(
                f"{self}: suffix-only conversation mode found no user message; "
                "sending full context"
            )
            return messages, False

        logger.debug(
            f"{self}: suffix-only conversation payload uses latest user message "
            f"with {self._count_audio_parts([latest_user])} audio parts"
        )
        return [latest_user], True

    def _latest_user_message(
        self,
        messages: list[dict[str, Any]],
        *,
        require_audio: bool,
    ) -> dict[str, Any] | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            if require_audio and self._count_audio_parts([message]) == 0:
                continue
            return copy.deepcopy(message)
        return None

    def _build_payload(self, audio_url: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self._settings.system_instruction:
            messages.append({"role": "system", "content": self._settings.system_instruction})

        content: list[dict[str, Any]] = []
        if self._settings.audio_prompt:
            content.append({"type": "text", "text": self._settings.audio_prompt})
        content.append({"type": "audio_url", "audio_url": {"url": audio_url}})
        messages.append({"role": "user", "content": content})

        return self._build_payload_from_messages(messages)

    def _build_payload_from_context_messages(self, context_messages: list[Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self._settings.system_instruction:
            messages.append({"role": "system", "content": self._settings.system_instruction})

        for message in context_messages:
            converted = self._convert_context_message(message)
            if converted:
                messages.append(converted)

        return self._build_payload_from_messages(messages)

    def _convert_context_message(self, message: Any) -> dict[str, Any] | None:
        if isinstance(message, LLMSpecificMessage):
            logger.debug(f"{self}: skipping LLM-specific context message for {message.llm}")
            return None
        if not isinstance(message, dict):
            logger.debug(f"{self}: skipping unsupported context message: {message!r}")
            return None

        role = message.get("role")
        if role == "developer":
            # vLLM's chat-completions path is OpenAI-compatible but the local
            # template support is model-dependent. Treat developer context as a
            # system instruction for broad compatibility.
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            logger.debug(f"{self}: skipping context message with unsupported role {role!r}")
            return None

        converted: dict[str, Any] = {"role": role}
        if "tool_calls" in message:
            converted["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            converted["tool_call_id"] = message["tool_call_id"]

        content = message.get("content")
        if isinstance(content, str) or content is None:
            converted["content"] = content
            return converted

        if not isinstance(content, list):
            logger.debug(f"{self}: skipping unsupported content in context message: {content!r}")
            return None

        converted_content: list[dict[str, Any]] = []
        for item in content:
            converted_item = self._convert_context_content_part(item)
            if converted_item:
                converted_content.append(converted_item)

        if not converted_content:
            return None

        converted["content"] = converted_content
        return converted

    def _convert_context_content_part(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")
        if item_type == "text":
            return {"type": "text", "text": item.get("text", "")}
        if item_type == "audio_url":
            return item
        if item_type == "input_audio":
            input_audio = item.get("input_audio") or {}
            data = input_audio.get("data") or item.get("audio")
            if not data:
                return None
            audio_format = input_audio.get("format") or item.get("format") or "wav"
            url = data if str(data).startswith("data:") else f"data:audio/{audio_format};base64,{data}"
            return {"type": "audio_url", "audio_url": {"url": url}}
        if item_type == "image_url":
            return item
        if "text" in item:
            return {"type": "text", "text": item["text"]}

        logger.debug(f"{self}: skipping unsupported context content part: {item!r}")
        return None

    def _count_audio_parts(self, messages: list[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            count += sum(1 for item in content if item.get("type") == "audio_url")
        return count

    async def _run_audio_completion(self, audio: bytes, sample_rate: int, num_channels: int):
        payload = self._build_payload(self._wav_data_url(audio, sample_rate, num_channels))
        await self._run_completion_payload(
            payload,
            request_description=(
                f"{len(audio)} bytes of {sample_rate} Hz audio "
                f"to {self._chat_completions_url}"
            ),
            start_ttfb=False,
        )

    async def _run_completion_payload(
        self,
        payload: dict[str, Any],
        *,
        request_description: str,
        start_ttfb: bool,
    ):
        started_at = time.perf_counter()
        first_token = True
        output_text_parts: list[str] = []
        tool_rounds = 0
        completed = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            if start_ttfb:
                await self.start_ttfb_metrics()

            headers = {"Accept": "text/event-stream"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            if not self._session:
                timeout = aiohttp.ClientTimeout(total=self._request_timeout_secs)
                self._session = aiohttp.ClientSession(timeout=timeout)

            cache_info = (
                f" with conversation_id={self._conversation_id}"
                if self._conversation_id
                else ""
            )
            logger.debug(f"{self}: sending {request_description}{cache_info}")

            current_payload = copy.deepcopy(payload)
            retried_full_context = False
            while True:
                try:
                    result = await self._stream_completion_pass(
                        current_payload,
                        headers=headers,
                        first_token=first_token,
                    )
                except ConversationCacheMissError:
                    if retried_full_context:
                        raise
                    full_payload = self._full_context_retry_payload(current_payload)
                    if full_payload is None:
                        raise
                    retried_full_context = True
                    current_payload = full_payload
                    logger.info(
                        f"{self}: conversation cache miss for "
                        f"{self._conversation_id}; retrying with full context"
                    )
                    continue
                first_token = result.first_token
                if result.output_text:
                    output_text_parts.append(result.output_text)
                if self._conversation_id:
                    self._conversation_cache_committed = True

                if not result.tool_calls:
                    break
                if not self._enable_bash_tool:
                    logger.warning(
                        f"{self}: model requested tool calls but bash tool is disabled"
                    )
                    break
                if tool_rounds >= self._bash_tool_max_rounds:
                    logger.warning(
                        f"{self}: reached bash tool round limit "
                        f"({self._bash_tool_max_rounds})"
                    )
                    break

                tool_rounds += 1
                logger.debug(
                    f"{self}: executing {len(result.tool_calls)} tool call(s) "
                    f"for round {tool_rounds}"
                )
                tool_messages = await self._execute_tool_calls(result.tool_calls)
                current_payload = self._payload_after_tool_calls(
                    current_payload,
                    result.tool_calls,
                    tool_messages,
                )

            completed = True
            logger.debug(
                f"{self}: completed response in {time.perf_counter() - started_at:.3f}s: "
                f"{''.join(output_text_parts)!r}"
            )
        except asyncio.CancelledError:
            logger.debug(f"{self}: audio completion cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: audio completion failed: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
        finally:
            if not completed and self._conversation_id:
                logger.debug(f"{self}: conversation cache commit unchanged after failed request")
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            if self._generation_task is asyncio.current_task():
                self._generation_task = None

    async def _stream_completion_pass(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        first_token: bool,
    ) -> ChatCompletionPassResult:
        output_text = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        async with self._session.post(
            self._chat_completions_url,
            json=self._http_payload(payload),
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                if self._is_conversation_cache_miss(response.status, error_text):
                    raise ConversationCacheMissError(error_text)
                raise RuntimeError(
                    f"vLLM request failed with {response.status}: {error_text}"
                )

            async for event in self._iter_sse_events(response):
                if event == "[DONE]":
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    logger.debug(f"{self}: skipping malformed SSE event: {event!r}")
                    continue

                usage = chunk.get("usage")
                if usage:
                    await self.start_llm_usage_metrics(
                        LLMTokenUsage(
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        )
                    )

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    for tool_call_delta in delta.get("tool_calls") or []:
                        self._merge_tool_call_delta(tool_calls_by_index, tool_call_delta)

                    text = delta.get("content") or ""
                    if not text:
                        continue
                    if first_token:
                        first_token = False
                        await self.stop_ttfb_metrics()
                    output_text += text
                    await self._push_llm_text(text)

        return ChatCompletionPassResult(
            output_text=output_text,
            tool_calls=self._finalize_tool_calls(tool_calls_by_index),
            first_token=first_token,
        )

    @staticmethod
    def _http_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if not key.startswith("_")
        }

    @staticmethod
    def _is_conversation_cache_miss(status: int, error_text: str) -> bool:
        if status != 409:
            return False
        try:
            data = json.loads(error_text)
        except json.JSONDecodeError:
            return False
        error = data.get("error")
        if not isinstance(error, dict):
            return False
        return error.get("type") == "ConversationCacheMissError"

    def _full_context_retry_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        full_messages = payload.get("_conversation_full_messages")
        if not isinstance(full_messages, list):
            logger.warning(
                f"{self}: cannot retry conversation cache miss without full messages"
            )
            return None

        retry_payload = copy.deepcopy(payload)
        retry_payload["messages"] = copy.deepcopy(full_messages)
        retry_payload.pop("conversation_require_cache", None)
        return retry_payload

    @staticmethod
    def _merge_tool_call_delta(
        tool_calls_by_index: dict[int, dict[str, Any]],
        tool_call_delta: dict[str, Any],
    ) -> None:
        index = tool_call_delta.get("index")
        if index is None:
            index = len(tool_calls_by_index)

        entry = tool_calls_by_index.setdefault(
            index,
            {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if tool_call_delta.get("id"):
            entry["id"] = tool_call_delta["id"]
        if tool_call_delta.get("type"):
            entry["type"] = tool_call_delta["type"]

        function_delta = tool_call_delta.get("function") or {}
        function = entry.setdefault("function", {"name": "", "arguments": ""})
        if function_delta.get("name"):
            function["name"] = f"{function.get('name') or ''}{function_delta['name']}"
        if function_delta.get("arguments"):
            function["arguments"] = (
                f"{function.get('arguments') or ''}{function_delta['arguments']}"
            )

    @staticmethod
    def _finalize_tool_calls(
        tool_calls_by_index: dict[int, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index):
            tool_call = copy.deepcopy(tool_calls_by_index[index])
            tool_call["id"] = tool_call.get("id") or f"call_{index}"
            tool_call["type"] = tool_call.get("type") or "function"
            function = tool_call.setdefault("function", {})
            function["name"] = function.get("name") or ""
            function["arguments"] = function.get("arguments") or "{}"
            tool_calls.append(tool_call)
        return tool_calls

    def _payload_after_tool_calls(
        self,
        payload: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        next_payload = copy.deepcopy(payload)
        next_payload["messages"] = [
            *copy.deepcopy(payload["messages"]),
            self._assistant_tool_call_message(tool_calls),
            *tool_messages,
        ]
        full_messages = next_payload.get("_conversation_full_messages")
        if isinstance(full_messages, list):
            next_payload["_conversation_full_messages"] = [
                *copy.deepcopy(full_messages),
                self._assistant_tool_call_message(tool_calls),
                *copy.deepcopy(tool_messages),
            ]
        next_payload.pop("tools", None)
        next_payload.pop("tool_choice", None)
        return next_payload

    @staticmethod
    def _assistant_tool_call_message(
        tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": copy.deepcopy(tool_calls),
        }

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tool_messages: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            result = await self._execute_tool_call(tool_call)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or "call_0",
                    "content": result,
                }
            )
        return tool_messages

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        if name != BASH_TOOL_NAME:
            return self._tool_result_json(
                ok=False,
                error=f"Unsupported tool: {name!r}",
            )

        arguments_text = function.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError as exc:
            return self._tool_result_json(
                ok=False,
                error=f"Invalid JSON arguments: {exc}",
                raw_arguments=arguments_text,
            )

        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return self._tool_result_json(
                ok=False,
                error="Missing required string argument: code",
                raw_arguments=arguments_text,
            )
        return await self._run_bash_tool(code, tool_call_id=tool_call.get("id") or "call_0")

    async def _run_bash_tool(self, code: str, *, tool_call_id: str) -> str:
        started_at = time.perf_counter()
        logger.debug(
            f"{self}: running bash tool in {self._bash_tool_cwd!r}: {code!r}"
        )
        await self._send_bash_tool_event(
            {
                "phase": "start",
                "tool_call_id": tool_call_id,
                "code": code,
                "cwd": self._bash_tool_cwd,
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                code,
                cwd=self._bash_tool_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            result = self._build_bash_tool_result(
                command=code,
                command_started=False,
                exit_code=None,
                stdout_text="",
                stderr_text="",
                timed_out=False,
                elapsed_secs=time.perf_counter() - started_at,
                error=f"Failed to start bash: {exc}",
            )
            self._log_bash_tool_result(tool_call_id=tool_call_id, code=code, result=result)
            await self._send_bash_tool_event(
                {
                    "phase": "result",
                    "tool_call_id": tool_call_id,
                    "code": code,
                    "cwd": self._bash_tool_cwd,
                    "result": result,
                }
            )
            return self._format_bash_tool_content(result)

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._bash_tool_timeout_secs,
            )
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        result = self._build_bash_tool_result(
            command=code,
            command_started=True,
            exit_code=process.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            timed_out=timed_out,
            elapsed_secs=time.perf_counter() - started_at,
        )
        self._log_bash_tool_result(tool_call_id=tool_call_id, code=code, result=result)
        await self._send_bash_tool_event(
            {
                "phase": "result",
                "tool_call_id": tool_call_id,
                "code": code,
                "cwd": self._bash_tool_cwd,
                "result": result,
            }
        )
        return self._format_bash_tool_content(result)

    def _format_bash_tool_content(self, result: dict[str, Any]) -> str:
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")

        if stdout and stderr:
            return f"<stdout>{stdout}</stdout>\n<stderr>{stderr}</stderr>"
        if stdout:
            return stdout
        if stderr:
            return stderr

        error = result.get("error")
        if error:
            return str(error)
        if result.get("timed_out"):
            return f"Command timed out after {self._bash_tool_timeout_secs:g} seconds."
        return "Command completed with no output."

    def _build_bash_tool_result(
        self,
        *,
        command: str,
        command_started: bool,
        exit_code: int | None,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool,
        elapsed_secs: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        stdout_truncated = len(stdout_text) > self._bash_tool_max_output_chars
        stderr_truncated = len(stderr_text) > self._bash_tool_max_output_chars
        command_not_found = command_started and exit_code == 127
        ok = command_started and exit_code == 0 and not timed_out and error is None

        if ok:
            status = "success"
            summary = "Command completed successfully."
            if stderr_text:
                summary += (
                    " The command wrote output to stderr, but exit_code is 0 so "
                    "that stderr output should not by itself be treated as failure."
                )
            assistant_guidance = (
                "The command succeeded. Answer from stdout and stderr. Do not say "
                "the command is missing or unrecognized."
            )
        elif timed_out:
            status = "timed_out"
            summary = f"Command timed out after {self._bash_tool_timeout_secs:g} seconds."
            assistant_guidance = (
                "The command timed out. Tell the user it did not finish and use any "
                "captured stdout or stderr if relevant."
            )
        elif not command_started:
            status = "failed_to_start"
            summary = error or "Command could not be started."
            assistant_guidance = "Bash could not start. Tell the user the tool failed to run."
        elif command_not_found:
            status = "command_not_found"
            summary = "Command exited with 127, which usually means the shell could not find it."
            assistant_guidance = "The command was not found. Tell the user it is missing or unavailable."
        else:
            status = "nonzero_exit"
            summary = (
                f"Command exited with status {exit_code}. Inspect stdout and stderr; "
                "a nonzero exit code can still include useful command output."
            )
            assistant_guidance = (
                "The command ran but exited nonzero. Use stdout and stderr as the "
                "result and mention the nonzero exit code if it matters."
            )

        result: dict[str, Any] = {
            "assistant_guidance": assistant_guidance,
            "command": command,
            "ok": ok,
            "status": status,
            "summary": summary,
            "command_started": command_started,
            "command_not_found": command_not_found,
            "exit_code": exit_code,
            "stdout": self._truncate_tool_output(stdout_text),
            "stderr": self._truncate_tool_output(stderr_text),
            "stdout_chars": len(stdout_text),
            "stderr_chars": len(stderr_text),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "timed_out": timed_out,
            "elapsed_secs": round(elapsed_secs, 3),
            "cwd": self._bash_tool_cwd,
        }
        if error:
            result["error"] = error
        return result

    def _log_bash_tool_result(
        self,
        *,
        tool_call_id: str,
        code: str,
        result: dict[str, Any],
    ) -> None:
        logger.debug(
            f"{self}: model-facing bash tool result for {tool_call_id} "
            f"({code!r}): {self._tool_result_json(**result)}"
        )

    async def _send_bash_tool_event(self, payload: dict[str, Any]) -> None:
        if not self._bash_tool_event_sender:
            return

        message = {
            "type": "bash-tool",
            "timestamp": time.time(),
            **payload,
        }
        try:
            await self._bash_tool_event_sender(message)
        except Exception as exc:
            logger.warning(f"{self}: failed to send bash tool RTVI event: {exc}")

    def _truncate_tool_output(self, text: str) -> str:
        if len(text) <= self._bash_tool_max_output_chars:
            return text
        omitted = len(text) - self._bash_tool_max_output_chars
        return (
            text[: self._bash_tool_max_output_chars]
            + f"\n...[truncated {omitted} chars]"
        )

    @staticmethod
    def _tool_result_json(**payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=True)

    async def _iter_sse_events(self, response: aiohttp.ClientResponse):
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                yield line[len("data:") :].strip()

        line = buffer.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()
