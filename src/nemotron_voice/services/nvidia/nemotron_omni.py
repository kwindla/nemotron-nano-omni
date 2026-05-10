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
import copy
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
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
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import NOT_GIVEN, LLMSettings, _NotGiven

_TRACE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

DEFAULT_VOICE_SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. Respond in plain text only. Keep answers "
    "brief, direct, and conversational, usually one or two short sentences. Your "
    "implementation uses the Nemotron Nano Omni LLM and the Kyutai Pocket TTS "
    "voice model. Your "
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
    "and say that the command rendered it as ASCII art. Only use the bash tool "
    "when the user's latest request explicitly asks you to inspect or operate on "
    "the local machine, run a command, or when local inspection is genuinely "
    "necessary to answer correctly. If the user explicitly asks you to use the "
    "bash tool, run a command, or report command output, you must call the bash "
    "tool and answer from that result rather than from memory. After you receive "
    "a bash tool result, do not call the same command again unless the user asks "
    "for a rerun or the situation has changed. Do not use the bash tool to echo, "
    "printf, paraphrase, or draft an answer you could simply say directly. The "
    "tool is for real command execution and explicit user-requested command "
    "output, not for generating prose. The user may ask about the Unix tool "
    "cowthink; use the bash tool to inspect it when needed."
)

BASH_TOOL_NAME = "run_bash"
DEFAULT_AUDIO_PROMPT = (
    "Listen to the audio and respond to the spoken instruction. If the user "
    "explicitly asks you to use the bash tool, run a command, inspect the local "
    "machine, or report command output, you must call run_bash and answer from "
    "the tool result. Otherwise answer directly without tools unless local "
    "inspection is genuinely needed. Never use bash just to echo or paraphrase "
    "an answer you could say directly."
)
NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE = 0.2
NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS = 1024
NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K = 1
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
            "you to inspect or operate on the local machine. If the user "
            "explicitly asks you to use bash, run a command, or report command "
            "output, call this tool instead of answering from memory. Do not call "
            "the exact same command again in the same assistant turn unless the "
            "tool result shows that a rerun is required. Do not use this tool to "
            "echo, printf, paraphrase, or draft natural-language answers that you "
            "could say directly. Examples: use "
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
        conversation_id: str | None = None,
        suffix_only_conversation: bool = True,
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
            conversation_id: Optional stable id sent to vLLM for exact
                conversation-cache reuse across turns.
            suffix_only_conversation: After the first successful cached turn,
                send only the latest user message for the conversation id.
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
            temperature=NEMOTRON_OMNI_INSTRUCT_DEFAULT_TEMPERATURE,
            max_tokens=NEMOTRON_OMNI_INSTRUCT_DEFAULT_MAX_TOKENS,
            top_p=None,
            top_k=NEMOTRON_OMNI_INSTRUCT_DEFAULT_TOP_K,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            audio_prompt=DEFAULT_AUDIO_PROMPT,
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
        self._conversation_id = conversation_id
        self._suffix_only_conversation = suffix_only_conversation
        self._request_timeout_secs = request_timeout_secs
        self._enable_bash_tool = enable_bash_tool
        self._bash_tool_cwd = bash_tool_cwd or os.getcwd()
        self._bash_tool_timeout_secs = bash_tool_timeout_secs
        self._bash_tool_max_output_chars = bash_tool_max_output_chars
        self._bash_tool_max_rounds = bash_tool_max_rounds
        self._bash_tool_event_sender = bash_tool_event_sender
        self._strip_historical_audio_from_payload = (
            os.getenv("NEMOTRON_OMNI_STRIP_HISTORICAL_AUDIO_FROM_PAYLOAD", "0") != "0"
        )

        self._session: aiohttp.ClientSession | None = None
        self._generation_task: asyncio.Task | None = None
        self._conversation_cache_committed = False
        self._canonical_messages: list[dict[str, Any]] = []
        self._context_lineage_messages: list[dict[str, Any]] = []
        trace_dir = os.getenv("NEMOTRON_OMNI_TRACE_DIR")
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_redact_audio = os.getenv("NEMOTRON_OMNI_TRACE_REDACT_AUDIO", "1") != "0"
        self._top_level_request_seq = 0

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
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._cancel_generation_task()
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

        canonical_messages = self._canonical_messages_from_context(messages)
        if canonical_messages is None:
            logger.debug(f"{self}: ignoring LLM context without a latest user message")
            return
        self._context_lineage_messages = copy.deepcopy(canonical_messages)

        await self._cancel_generation_task()
        payload, full_messages = self._build_payload_from_messages(canonical_messages)
        self._generation_task = self.create_task(
            self._run_completion_payload(
                payload,
                full_messages=full_messages,
                request_description=(
                    f"context with {len(payload['messages'])} messages and "
                    f"{self._count_audio_parts(payload['messages'])} audio parts"
                ),
                start_ttfb=True,
            ),
            name="nemotron_omni_context_completion",
        )

    async def _cancel_generation_task(self):
        if self._generation_task:
            await self.cancel_task(self._generation_task)
            self._generation_task = None

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _build_payload_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        full_messages = copy.deepcopy(messages)
        if self._strip_historical_audio_from_payload:
            full_messages = self._strip_historical_audio_from_messages(full_messages)
        payload_messages, requires_cache = self._conversation_payload_messages(full_messages)
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": payload_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
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

        return payload, full_messages

    def _strip_historical_audio_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest_user_index = -1
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                latest_user_index = index
                break
        if latest_user_index < 0:
            return copy.deepcopy(messages)

        stripped_messages = copy.deepcopy(messages)
        stripped_audio_parts = 0
        for index, message in enumerate(stripped_messages):
            if index == latest_user_index or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            filtered_content = []
            for item in content:
                if not isinstance(item, dict):
                    filtered_content.append(item)
                    continue
                item_type = item.get("type")
                if item_type in {"audio_url", "input_audio"}:
                    stripped_audio_parts += 1
                    continue
                filtered_content.append(item)
            if filtered_content:
                message["content"] = filtered_content

        if stripped_audio_parts:
            logger.debug(
                f"{self}: stripped {stripped_audio_parts} historical audio part(s) "
                "from payload messages"
            )
        return stripped_messages

    def _conversation_payload_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if (
            not self._conversation_id
            or not self._suffix_only_conversation
            or not self._conversation_cache_committed
        ):
            return copy.deepcopy(messages), False

        latest_user = self._latest_user_message(messages)
        if latest_user is None:
            logger.warning(
                f"{self}: suffix-only conversation mode found no user message; "
                "sending full context"
            )
            return copy.deepcopy(messages), False

        logger.debug(
            f"{self}: suffix-only conversation payload uses latest user message "
            f"with {self._count_audio_parts([latest_user])} audio parts"
        )
        return [latest_user], True

    def _latest_user_message(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            return copy.deepcopy(message)
        return None

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
        if "name" in message:
            converted["name"] = message["name"]

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

    def _message_role_summary(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "?")
            audio_parts = self._count_audio_parts([message])
            if audio_parts:
                parts.append(f"{role}[audio={audio_parts}]")
            elif message.get("tool_calls"):
                parts.append(f"{role}[tool_calls]")
            elif role == "tool":
                parts.append(f"{role}[result]")
            else:
                parts.append(role)
        return ",".join(parts)

    def _canonical_messages_from_context(
        self,
        context_messages: list[Any],
    ) -> list[dict[str, Any]] | None:
        converted_messages: list[dict[str, Any]] = []
        for message in context_messages:
            converted = self._convert_context_message(message)
            if converted is not None:
                converted_messages.append(converted)

        canonical_messages = self._with_system_message(converted_messages)
        if (
            not canonical_messages
            or canonical_messages[-1].get("role") != "user"
            or self._latest_user_message(canonical_messages) is None
        ):
            return None

        if not self._context_messages_extend_observed_lineage(canonical_messages):
            self._rotate_conversation_cache_lineage(canonical_messages)
        return canonical_messages

    def _context_messages_extend_observed_lineage(
        self,
        observed_messages: list[dict[str, Any]],
    ) -> bool:
        lineage_count = len(self._context_lineage_messages)
        if lineage_count == 0:
            return True
        if len(observed_messages) < lineage_count:
            return False
        return observed_messages[:lineage_count] == self._context_lineage_messages

    def _rotate_conversation_cache_lineage(
        self,
        observed_messages: list[dict[str, Any]],
    ) -> None:
        old_conversation_id = self._conversation_id
        if self._conversation_id:
            self._conversation_id = f"pipecat-{uuid.uuid4().hex}"
            logger.info(
                f"{self}: rotating conversation_id from {old_conversation_id} "
                f"to {self._conversation_id} after non-append context change"
            )
        else:
            logger.info(
                f"{self}: resetting uncached conversation lineage after "
                "non-append context change"
            )
        logger.debug(
            f"{self}: committed roles={self._message_role_summary(self._context_lineage_messages)} "
            f"observed roles={self._message_role_summary(observed_messages)}"
        )
        self._conversation_cache_committed = False
        self._canonical_messages = []
        self._context_lineage_messages = []

    def _commit_canonical_messages(
        self,
        full_messages: list[dict[str, Any]],
        *,
        assistant_text: str,
    ) -> None:
        committed_messages = self._with_system_message(full_messages)
        if assistant_text:
            committed_messages.append({"role": "assistant", "content": assistant_text})
        self._canonical_messages = committed_messages

    def _with_system_message(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_messages = copy.deepcopy(messages)
        if not self._settings.system_instruction:
            return normalized_messages

        system_message = {
            "role": "system",
            "content": self._settings.system_instruction,
        }
        if normalized_messages and normalized_messages[0].get("role") == "system":
            normalized_messages[0] = system_message
            return normalized_messages
        return [system_message, *normalized_messages]

    def _trace_request_id(self, top_level_request_seq: int, attempt_num: int) -> str:
        conversation_part = self._conversation_id or "no-conversation"
        return (
            f"nemotron-{conversation_part}-turn-{top_level_request_seq:03d}-"
            f"attempt-{attempt_num:02d}"
        )

    def _trace_json_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if (
                self._trace_redact_audio
                and
                set(value.keys()) == {"url"}
                and isinstance(value["url"], str)
                and value["url"].startswith("data:audio/")
            ):
                url = value["url"]
                _, _, encoded = url.partition(",")
                digest = hashlib.sha256(encoded.encode("ascii")).hexdigest()
                return {
                    "url": f"<data-audio-base64 sha256={digest} chars={len(encoded)}>",
                }
            return {
                str(key): self._trace_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._trace_json_value(item) for item in value]
        return value

    def _write_trace_file(
        self,
        *,
        trace_id: str,
        phase: str,
        payload: dict[str, Any],
    ) -> None:
        if self._trace_dir is None:
            return
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            safe_trace_id = _TRACE_NAME_RE.sub("_", Path(str(trace_id)).name)
            safe_phase = _TRACE_NAME_RE.sub("_", Path(str(phase)).name)
            if not safe_trace_id or not safe_phase:
                return
            trace_path = self._trace_dir / f"{safe_trace_id}.{safe_phase}.json"
            trace_payload = {
                "trace_id": trace_id,
                "phase": phase,
                "timestamp": time.time(),
                **payload,
            }
            trace_path.write_text(
                json.dumps(
                    self._trace_json_value(trace_payload),
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"{self}: failed to write trace file for {trace_id}: {exc}")

    async def _run_completion_payload(
        self,
        payload: dict[str, Any],
        *,
        full_messages: list[dict[str, Any]],
        request_description: str,
        start_ttfb: bool,
    ):
        started_at = time.perf_counter()
        first_token = True
        output_text_parts: list[str] = []
        tool_rounds = 0
        completed = False
        recovered_from_cache_miss = False
        seen_tool_results_by_signature: dict[tuple[str, str], str] = {}
        final_assistant_text = ""
        self._top_level_request_seq += 1
        top_level_request_seq = self._top_level_request_seq

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
            current_full_messages = copy.deepcopy(full_messages)
            retried_full_context = False
            attempt_num = 0
            while True:
                attempt_num += 1
                trace_id = self._trace_request_id(top_level_request_seq, attempt_num)
                attempt_messages = current_payload.get("messages")
                if not isinstance(attempt_messages, list):
                    attempt_messages = []
                logger.debug(
                    f"{self}: completion attempt {attempt_num} "
                    f"messages={len(attempt_messages)} "
                    f"roles={self._message_role_summary(attempt_messages)} "
                    f"audio_parts={self._count_audio_parts(attempt_messages)} "
                    f"require_cache={bool(current_payload.get('conversation_require_cache'))}"
                )
                http_payload = self._http_payload(current_payload)
                self._write_trace_file(
                    trace_id=trace_id,
                    phase="client-request",
                    payload={
                        "request_description": request_description,
                        "conversation_id": self._conversation_id,
                        "suffix_only_conversation": self._suffix_only_conversation,
                        "conversation_cache_committed": self._conversation_cache_committed,
                        "messages_role_summary": self._message_role_summary(
                            attempt_messages
                        ),
                        "http_payload": http_payload,
                        "conversation_full_messages": current_full_messages,
                    },
                )
                try:
                    result = await self._stream_completion_pass(
                        current_payload,
                        headers={**headers, "X-Request-Id": trace_id},
                        first_token=first_token,
                        trace_id=trace_id,
                    )
                except ConversationCacheMissError:
                    if retried_full_context:
                        raise
                    full_payload = self._full_context_retry_payload(
                        current_payload,
                        current_full_messages,
                    )
                    recovered_from_cache_miss = True
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

                if not result.tool_calls:
                    final_assistant_text = result.output_text
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
                tool_messages = await self._execute_tool_calls(
                    result.tool_calls,
                    seen_tool_results_by_signature=seen_tool_results_by_signature,
                )
                current_payload, current_full_messages = self._payload_after_tool_calls(
                    current_payload,
                    current_full_messages,
                    assistant_text=result.output_text,
                    tool_calls=result.tool_calls,
                    tool_messages=tool_messages,
                )

            completed = True
            self._commit_canonical_messages(
                current_full_messages,
                assistant_text=final_assistant_text,
            )
            if self._conversation_id:
                self._conversation_cache_committed = not recovered_from_cache_miss
                if recovered_from_cache_miss:
                    logger.debug(
                        f"{self}: keeping suffix-only conversation mode disabled "
                        "for the next turn after cache-miss recovery"
                    )
            logger.debug(
                f"{self}: completed response in {time.perf_counter() - started_at:.3f}s: "
                f"{''.join(output_text_parts)!r}"
            )
        except asyncio.CancelledError:
            logger.debug(f"{self}: completion cancelled")
            raise
        except Exception as e:
            logger.error(f"{self}: completion failed: {e}")
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
        trace_id: str,
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
                self._write_trace_file(
                    trace_id=trace_id,
                    phase="client-error",
                    payload={
                        "status": response.status,
                        "response_text": error_text,
                    },
                )
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
                    tool_call_deltas = delta.get("tool_calls") or []
                    for tool_call_delta in tool_call_deltas:
                        self._merge_tool_call_delta(tool_calls_by_index, tool_call_delta)

                    text = delta.get("content") or ""
                    if not text:
                        continue
                    if first_token:
                        first_token = False
                        await self.stop_ttfb_metrics()
                    output_text += text
                    await self._push_llm_text(text)

        result = ChatCompletionPassResult(
            output_text=output_text,
            tool_calls=self._finalize_tool_calls(tool_calls_by_index),
            first_token=first_token,
        )
        self._write_trace_file(
            trace_id=trace_id,
            phase="client-response",
            payload={
                "output_text": result.output_text,
                "tool_calls": result.tool_calls,
                "first_token_pending": result.first_token,
            },
        )
        return result

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
        full_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
        full_messages: list[dict[str, Any]],
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        tool_messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assistant_tool_call_message = self._assistant_tool_call_message(
            tool_calls,
            assistant_text=assistant_text,
        )
        next_payload = copy.deepcopy(payload)
        next_full_messages = [
            *copy.deepcopy(full_messages),
            assistant_tool_call_message,
            *copy.deepcopy(tool_messages),
        ]
        if self._conversation_id:
            # vLLM's frontend ledger reconstructs the assistant tool-call
            # message. The engine cache physically commits only a prompt-prefix
            # checkpoint, so send just the tool-result suffix and require cache
            # to avoid duplicating the same top-level turn.
            next_payload["messages"] = copy.deepcopy(tool_messages)
            next_payload["conversation_require_cache"] = True
        else:
            next_payload["messages"] = [
                *copy.deepcopy(payload["messages"]),
                assistant_tool_call_message,
                *copy.deepcopy(tool_messages),
            ]
        # Keep tool definitions stable across tool-followup requests. vLLM feeds
        # `tools` into the chat template, so removing them shrinks the prompt and
        # breaks exact conversation-cache attach on the next round.
        return next_payload, next_full_messages

    @staticmethod
    def _assistant_tool_call_message(
        tool_calls: list[dict[str, Any]],
        *,
        assistant_text: str,
    ) -> dict[str, Any]:
        content: str | None = assistant_text or None
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": copy.deepcopy(tool_calls),
        }

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        seen_tool_results_by_signature: dict[tuple[str, str], str] | None = None,
    ) -> list[dict[str, Any]]:
        tool_messages: list[dict[str, Any]] = []
        signature_results = seen_tool_results_by_signature or {}
        for tool_call in tool_calls:
            signature = self._tool_call_signature(tool_call)
            if signature and signature in signature_results:
                result = self._duplicate_tool_result(signature_results[signature])
                logger.debug(
                    f"{self}: suppressing duplicate tool call within one user turn: "
                    f"{signature[0]} {signature[1]!r}"
                )
                await self._send_bash_tool_event(
                    {
                        "phase": "duplicate_suppressed",
                        "guardrail_triggered": True,
                        "guardrail_kind": "duplicate_tool_call",
                        "guardrail_reason": "exact_duplicate_command",
                        "tool_call_id": tool_call.get("id") or "call_0",
                        "name": signature[0],
                        "signature": signature[1],
                    }
                )
            else:
                result = await self._execute_tool_call(tool_call)
                if signature:
                    signature_results[signature] = result
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or "call_0",
                    "name": (tool_call.get("function") or {}).get("name") or "",
                    "content": result,
                }
            )
        return tool_messages

    @staticmethod
    def _tool_call_signature(
        tool_call: dict[str, Any]
    ) -> tuple[str, str] | None:
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        if name != BASH_TOOL_NAME:
            return None

        arguments_text = function.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError:
            return None

        code = arguments.get("code")
        if not isinstance(code, str):
            return None

        normalized_code = code.strip()
        if not normalized_code:
            return None
        return name, normalized_code

    @staticmethod
    def _duplicate_tool_result(previous_result: str) -> str:
        return (
            "[tool policy] Exact duplicate tool call suppressed. "
            "Reuse the prior result below instead of calling the same command "
            "again.\n"
            f"{previous_result}"
        )

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
        if self._is_prose_echo_command(code):
            logger.debug(
                f"{self}: suppressing non-instrumental bash tool call: {code!r}"
            )
            await self._send_bash_tool_event(
                {
                    "phase": "policy_rejected",
                    "guardrail_triggered": True,
                    "guardrail_kind": "non_instrumental_bash_tool_use",
                    "guardrail_reason": "echo_or_printf_prose",
                    "tool_call_id": tool_call.get("id") or "call_0",
                    "name": name,
                    "code": code,
                }
            )
            return (
                "[tool policy] Do not use bash to echo, printf, or paraphrase an "
                "answer you could say directly. The bash tool is only for real "
                "command execution or explicit user-requested command output. "
                "Answer the user directly without further tool use."
            )
        return await self._run_bash_tool(code, tool_call_id=tool_call.get("id") or "call_0")

    @staticmethod
    def _is_prose_echo_command(code: str) -> bool:
        stripped = code.strip()
        if not stripped or any(token in stripped for token in ("&&", "||", "|", ";", "$(", "`")):
            return False
        try:
            argv = shlex.split(stripped)
        except ValueError:
            return False
        if not argv or argv[0] not in {"echo", "printf"}:
            return False
        payload_tokens = argv[1:]
        if not payload_tokens:
            return False
        payload_text = " ".join(token.rstrip("\\n") for token in payload_tokens).strip()
        if not payload_text:
            return False
        word_count = len(re.findall(r"[A-Za-z0-9]+", payload_text))
        has_sentence_punctuation = any(char in payload_text for char in ".!,?:;")
        return word_count >= 12 or (word_count >= 8 and has_sentence_punctuation)

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
