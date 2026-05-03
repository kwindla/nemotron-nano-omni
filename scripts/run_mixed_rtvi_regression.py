#!/usr/bin/env python3
"""Run a mixed 20-turn SmallWebRTC regression and validate cache behavior."""

from __future__ import annotations

import argparse
import ast
import asyncio
import audioop
import json
import re
import time
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame
from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = ROOT / "media" / "cartesia-regression"
DEFAULT_BOT_LOG = ROOT / "logs" / "bot.log"
DEFAULT_VLLM_LOG = ROOT / "logs" / "dgx-spark-vllm.log"
CONVERSATION_ID_MARKER = "Using Nemotron Omni conversation_id="
REPO_ROOT_TOKEN = str(ROOT)


@dataclass(frozen=True)
class Action:
    label: str
    kind: str
    content: str
    expect_substrings: tuple[str, ...]
    audio_response: bool = False
    audio_path: Path | None = None


DEFAULT_ACTIONS = [
    Action(
        label="audio-unicorn",
        kind="audio",
        content="audio_unicorn_intro.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_unicorn_intro.wav",
        expect_substrings=("unicorn",),
        audio_response=True,
    ),
    Action(
        label="text-recall-unicorn",
        kind="text",
        content="What creature did I mention in the previous audio? Answer with one word only.",
        expect_substrings=("unicorn",),
        audio_response=True,
    ),
    Action(
        label="text-tool-pwd",
        kind="text",
        content=(
            "You must use the bash tool now. Run exactly this command: pwd. "
            "Do not answer from memory or prior knowledge. Reply with only the tool stdout."
        ),
        expect_substrings=("nemotron-nano-omni",),
        audio_response=True,
    ),
    Action(
        label="text-followup-pwd",
        kind="text",
        content="What exact path did the previous command print? Reply with the path only.",
        expect_substrings=(REPO_ROOT_TOKEN,),
        audio_response=True,
    ),
    Action(
        label="audio-tool-echo-one",
        kind="audio",
        content="audio_tool_echo_one.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_tool_echo_one.wav",
        expect_substrings=("audio1|audio 1|spark audio one|spark_audio_1",),
        audio_response=True,
    ),
    Action(
        label="text-followup-echo-one",
        kind="text",
        content="What exact words did the previous spoken command print? Reply with those words only.",
        expect_substrings=("audio1|audio 1|spark audio one|spark_audio_1",),
        audio_response=True,
    ),
    Action(
        label="audio-dragon",
        kind="audio",
        content="audio_dragon_intro.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_dragon_intro.wav",
        expect_substrings=("dragon",),
        audio_response=True,
    ),
    Action(
        label="text-recall-dragon",
        kind="text",
        content="What creature did I mention in the previous audio? Answer with one word only.",
        expect_substrings=("dragon",),
        audio_response=True,
    ),
    Action(
        label="text-tool-echo-two",
        kind="text",
        content=(
            "You must use the bash tool now. Run exactly this command: "
            "echo spark text two. Do not answer from memory or prior knowledge. "
            "Reply with only the tool stdout."
        ),
        expect_substrings=("spark text two",),
        audio_response=True,
    ),
    Action(
        label="text-followup-echo-two",
        kind="text",
        content="What exact words did the previous command print? Reply with those words only.",
        expect_substrings=("spark text two",),
        audio_response=True,
    ),
    Action(
        label="audio-tool-echo-three",
        kind="audio",
        content="audio_tool_echo_three.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_tool_echo_three.wav",
        expect_substrings=("audio3|audio 3|spark audio three|spark_audio_3",),
        audio_response=True,
    ),
    Action(
        label="text-followup-echo-three",
        kind="text",
        content="What exact words did the previous spoken command print? Reply with those words only.",
        expect_substrings=("audio3|audio 3|spark audio three|spark_audio_3",),
        audio_response=True,
    ),
    Action(
        label="text-recall-dragon-again",
        kind="text",
        content="What creature was mentioned in the most recent creature-themed audio turn? Answer with one word only.",
        expect_substrings=("dragon",),
        audio_response=True,
    ),
    Action(
        label="audio-math",
        kind="audio",
        content="audio_math_1000_div_25.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_math_1000_div_25.wav",
        expect_substrings=("40",),
        audio_response=True,
    ),
    Action(
        label="text-followup-math",
        kind="text",
        content="Repeat just the number from the previous audio answer.",
        expect_substrings=("40",),
        audio_response=True,
    ),
    Action(
        label="text-tool-echo-four",
        kind="text",
        content=(
            "You must use the bash tool now. Run exactly this command: "
            "echo spark text four. Do not answer from memory or prior knowledge. "
            "Reply with only the tool stdout."
        ),
        expect_substrings=("spark text four",),
        audio_response=True,
    ),
    Action(
        label="text-followup-echo-four",
        kind="text",
        content="What exact words did the previous command print? Reply with those words only.",
        expect_substrings=("spark text four",),
        audio_response=True,
    ),
    Action(
        label="audio-goodbye",
        kind="audio",
        content="audio_goodbye.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_goodbye.wav",
        expect_substrings=("goodbye",),
        audio_response=True,
    ),
    Action(
        label="text-followup-goodbye",
        kind="text",
        content="What was the last word I asked you to say? Reply with one word only.",
        expect_substrings=("goodbye",),
        audio_response=True,
    ),
    Action(
        label="audio-tool-echo-five",
        kind="audio",
        content="audio_tool_echo_five.wav",
        audio_path=DEFAULT_AUDIO_DIR / "audio_tool_echo_five.wav",
        expect_substrings=(
            "finalaudiofive|final audio5|audio5|audio 5|final audio five|final audio 5|final_audio_five|final_audio_5",
        ),
        audio_response=True,
    ),
]


class QueuedWavTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        frame_ms: int = 20,
        turn_gap_secs: float = 2.0,
    ):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = int(sample_rate * frame_ms / 1000)
        self._frame_duration = frame_ms / 1000
        self._bytes_per_frame = self._samples_per_frame * 2
        self._turn_gap_frames = max(1, int(turn_gap_secs / self._frame_duration))
        self._queue: asyncio.Queue[tuple[str, bytes, asyncio.Event]] = asyncio.Queue()
        self._current_label: str | None = None
        self._current_pcm = b""
        self._current_pos = 0
        self._current_done: asyncio.Event | None = None
        self._gap_frames_remaining = 0
        self._pts = 0

    @staticmethod
    def _load_wav(path: Path, target_sample_rate: int) -> bytes:
        with wave.open(str(path), "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            pcm = audioop.lin2lin(pcm, sample_width, 2)
            sample_width = 2
        if num_channels != 1:
            pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
            num_channels = 1
        if sample_rate != target_sample_rate:
            pcm, _ = audioop.ratecv(
                pcm, sample_width, num_channels, sample_rate, target_sample_rate, None
            )
        return pcm

    async def enqueue(self, label: str, path: Path) -> asyncio.Event:
        done = asyncio.Event()
        await self._queue.put((label, self._load_wav(path, self._sample_rate), done))
        return done

    def _next_chunk(self) -> bytes:
        if self._gap_frames_remaining > 0:
            self._gap_frames_remaining -= 1
            return b"\x00" * self._bytes_per_frame

        if self._current_label is None:
            try:
                label, pcm, done = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return b"\x00" * self._bytes_per_frame
            self._current_label = label
            self._current_pcm = pcm
            self._current_pos = 0
            self._current_done = done
            logger.info(f"Start audio turn {label}")

        chunk = self._current_pcm[self._current_pos : self._current_pos + self._bytes_per_frame]
        self._current_pos += len(chunk)
        if len(chunk) < self._bytes_per_frame:
            chunk += b"\x00" * (self._bytes_per_frame - len(chunk))

        if self._current_pos >= len(self._current_pcm):
            logger.info(f"Finished audio turn {self._current_label}")
            if self._current_done is not None:
                self._current_done.set()
            self._current_label = None
            self._current_pcm = b""
            self._current_pos = 0
            self._current_done = None
            self._gap_frames_remaining = self._turn_gap_frames

        return chunk

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(self._frame_duration)
        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.planes[0].update(self._next_chunk())
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += self._samples_per_frame
        return frame


@dataclass
class PendingTurn:
    action: Action
    llm_started: asyncio.Event = field(default_factory=asyncio.Event)
    tts_started: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    llm_text: list[str] = field(default_factory=list)
    tts_text: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def response_text(self) -> str:
        if self.tts_text:
            return "".join(self.tts_text)
        return "".join(self.llm_text)


def _rtvi_message(msg_type: str, data: dict[str, Any] | None = None) -> str:
    payload = {"label": "rtvi-ai", "type": msg_type, "id": f"rtvi-{time.time_ns()}"}
    if data is not None:
        payload["data"] = data
    return json.dumps(payload)


async def _wait_for_ice_gathering(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


def _read_new_text(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        return handle.read()


def _file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _missing_expected_response_substrings(
    action: Action,
    response_text: str,
) -> list[str]:
    normalized = _normalize(response_text)
    missing: list[str] = []
    for value in action.expect_substrings:
        alternatives = [_normalize(item) for item in value.split("|")]
        if not any(item and item in normalized for item in alternatives):
            missing.append(value)
    return missing


def _assert_expected_response(action: Action, response_text: str) -> None:
    missing = _missing_expected_response_substrings(action, response_text)
    if missing:
        raise AssertionError(
            f"{action.label} response missing expected substring(s) {missing}: {response_text!r}"
        )


def _extract_completed_responses(bot_log_text: str) -> list[str]:
    responses: list[str] = []
    for raw_line in bot_log_text.splitlines():
        if "completed response in" not in raw_line:
            continue
        _, _, suffix = raw_line.partition("completed response in")
        _, _, literal = suffix.partition(": ")
        literal = literal.strip()
        if not literal:
            continue
        try:
            value = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            value = literal
        responses.append(str(value))
    return responses


def _extract_latest_conversation_id(bot_log_text: str) -> str | None:
    matches = re.findall(rf"{re.escape(CONVERSATION_ID_MARKER)}([^\s]+)", bot_log_text)
    if matches:
        return matches[-1]
    return None


def _conversation_log_window(log_text: str, conversation_id: str | None) -> str:
    if not conversation_id:
        return log_text

    marker = f"{CONVERSATION_ID_MARKER}{conversation_id}"
    start = log_text.rfind(marker)
    if start == -1:
        return log_text

    next_start = log_text.find(CONVERSATION_ID_MARKER, start + len(marker))
    if next_start == -1:
        return log_text[start:]
    return log_text[start:next_start]


def _select_bot_analysis_text(
    *,
    bot_text: str,
    full_bot_log: str,
    conversation_id: str | None,
) -> tuple[str, str]:
    if re.search(r"completion attempt \d+", bot_text):
        return bot_text, "slice"

    fallback = _conversation_log_window(full_bot_log, conversation_id)
    if re.search(r"completion attempt \d+", fallback):
        return fallback, "full_conversation_window"
    return bot_text, "slice"


def _select_vllm_analysis_text(
    *,
    vllm_text: str,
    full_vllm_log: str,
    conversation_id: str | None,
) -> tuple[str, str]:
    if conversation_id and conversation_id in vllm_text:
        return vllm_text, "slice"
    if conversation_id and conversation_id in full_vllm_log:
        return full_vllm_log, "full_log"
    return vllm_text, "slice"


def _validate_logs(
    *,
    bot_text: str,
    vllm_text: str,
    conversation_id: str | None,
    expect_cache_attach: str,
) -> dict[str, int]:
    failure_patterns = [
        ("bot conversation cache miss", r"conversation cache miss"),
        ("bot already generating", r"already generating"),
        ("vllm 409", r"409 Conflict"),
        ("vllm attach skipped", r"attach skipped"),
        ("vllm cache miss error", r"ConversationCacheMissError"),
    ]
    for label, pattern in failure_patterns:
        if re.search(pattern, bot_text, flags=re.IGNORECASE) or re.search(
            pattern, vllm_text, flags=re.IGNORECASE
        ):
            raise AssertionError(f"Found forbidden log pattern: {label}")

    require_cache_attempts = len(
        re.findall(r"completion attempt \d+ .*require_cache=True", bot_text)
    )
    attach_count = 0
    if conversation_id:
        attach_count = len(
            re.findall(
                rf"Attached conversation cache for {re.escape(conversation_id)} ",
                vllm_text,
            )
        )
        if expect_cache_attach == "always" and require_cache_attempts != attach_count:
            raise AssertionError(
                "Cache attach count mismatch: "
                f"require_cache_attempts={require_cache_attempts}, attach_count={attach_count}"
            )
        if expect_cache_attach == "never" and attach_count != 0:
            raise AssertionError(
                "Expected cache attaches to stay disabled, "
                f"found attach_count={attach_count}"
            )

    return {
        "require_cache_attempts": require_cache_attempts,
        "attach_count": attach_count,
        "bot_completion_attempts": len(re.findall(r"completion attempt \d+", bot_text)),
    }


async def run(args: argparse.Namespace) -> None:
    actions = list(DEFAULT_ACTIONS)
    if args.max_turns is not None:
        if args.max_turns <= 0:
            raise RuntimeError("--max-turns must be positive when provided")
        actions = actions[: args.max_turns]
    missing = [action.audio_path for action in actions if action.audio_path and not action.audio_path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise RuntimeError(
            "Missing audio fixtures:\n"
            f"{missing_text}\n"
            "Run scripts/generate_cartesia_audio_fixtures.py first."
        )

    bot_log_offset = _file_size(args.bot_log)
    vllm_log_offset = _file_size(args.vllm_log)

    pc = RTCPeerConnection()
    track = QueuedWavTrack(turn_gap_secs=args.turn_gap_secs)
    channel = pc.createDataChannel("rtvi-ai")
    channel_open = asyncio.Event()
    bot_ready = asyncio.Event()
    pending_turn: PendingTurn | None = None
    pending_lock = asyncio.Lock()
    received_message_types: list[str] = []
    live_action_responses: list[dict[str, str]] = []
    response_mismatches: list[dict[str, Any]] = []

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track_obj) -> None:
        logger.info(f"Receiving remote {track_obj.kind} track")

        async def receive() -> None:
            got_first_frame = False
            while True:
                try:
                    await track_obj.recv()
                except Exception as exc:  # pragma: no cover - transport cleanup
                    logger.info(f"Remote {track_obj.kind} track ended: {exc}")
                    return
                if not got_first_frame:
                    got_first_frame = True
                    logger.info(f"Received first remote {track_obj.kind} frame")

        asyncio.create_task(receive())

    @channel.on("open")
    def on_open() -> None:
        logger.info("RTVI data channel open")
        channel.send(
            _rtvi_message(
                "client-ready",
                {
                    "version": "1.2.0",
                    "about": {
                        "library": "mixed-regression",
                        "library_version": "1",
                        "platform": "python",
                    },
                },
            )
        )
        channel_open.set()

    @channel.on("message")
    def on_message(message: str | bytes) -> None:
        nonlocal pending_turn
        if not isinstance(message, str):
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.info(f"RTVI non-JSON message: {message}")
            return

        msg_type = payload.get("type", "<missing>")
        received_message_types.append(msg_type)

        if msg_type == "bot-ready":
            bot_ready.set()
            return

        if msg_type in {"error", "error-response"}:
            error_text = json.dumps(payload, sort_keys=True)
            logger.error(error_text)
            if pending_turn is not None:
                pending_turn.errors.append(error_text)
                pending_turn.done.set()
            return

        if pending_turn is None:
            return

        if msg_type == "bot-llm-started":
            pending_turn.llm_text.clear()
            pending_turn.tts_text.clear()
            pending_turn.llm_started.set()
            return

        if msg_type == "bot-llm-text":
            if not pending_turn.llm_started.is_set():
                return
            text = ((payload.get("data") or {}).get("text")) or ""
            if text:
                pending_turn.llm_text.append(text)
            return

        if msg_type == "bot-tts-started":
            pending_turn.tts_text.clear()
            pending_turn.tts_started.set()
            return

        if msg_type == "bot-tts-text":
            if not pending_turn.tts_started.is_set():
                return
            text = ((payload.get("data") or {}).get("text")) or ""
            if text:
                pending_turn.tts_text.append(text)
            return

        if (
            msg_type == "bot-llm-stopped"
            and not pending_turn.action.audio_response
            and pending_turn.llm_started.is_set()
        ):
            pending_turn.done.set()
            return

        if (
            msg_type == "bot-tts-stopped"
            and pending_turn.action.audio_response
            and pending_turn.tts_started.is_set()
        ):
            pending_turn.done.set()
            return

    pc.addTrack(track)
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _wait_for_ice_gathering(pc)

    payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    logger.info(f"Posting offer to {args.offer_url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(args.offer_url, json=payload) as response:
            response.raise_for_status()
            answer = await response.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    try:
        await asyncio.wait_for(channel_open.wait(), timeout=args.connect_timeout)
        await asyncio.wait_for(bot_ready.wait(), timeout=args.connect_timeout)

        for index, action in enumerate(actions, start=1):
            logger.info(f"Turn {index}/{len(actions)} {action.label}")
            async with pending_lock:
                pending_turn = PendingTurn(action=action)

            if action.kind == "text":
                channel.send(
                    _rtvi_message(
                        "send-text",
                        {
                            "content": action.content,
                            "options": {
                                "run_immediately": True,
                                "audio_response": action.audio_response,
                            },
                        },
                    )
                )
            else:
                assert action.audio_path is not None
                send_done = await track.enqueue(action.label, action.audio_path)
                await asyncio.wait_for(send_done.wait(), timeout=args.audio_send_timeout)

            await asyncio.wait_for(pending_turn.llm_started.wait(), timeout=args.turn_timeout)
            if action.audio_response:
                await asyncio.wait_for(
                    pending_turn.tts_started.wait(),
                    timeout=args.turn_timeout,
                )
            await asyncio.wait_for(pending_turn.done.wait(), timeout=args.turn_timeout)
            if pending_turn.errors:
                raise AssertionError(
                    f"{action.label} received RTVI error(s): {pending_turn.errors}"
                )

            response_text = pending_turn.response_text()
            missing = _missing_expected_response_substrings(action, response_text)
            if missing:
                mismatch = {
                    "label": action.label,
                    "response": response_text,
                    "missing_expected_substrings": missing,
                    "source": "live_rtvi",
                }
                if not args.allow_response_mismatches:
                    raise AssertionError(
                        f"{action.label} response missing expected substring(s) "
                        f"{missing}: {response_text!r}"
                    )
                logger.warning(
                    "{} response mismatch {}: {!r}",
                    action.label,
                    missing,
                    response_text,
                )
                response_mismatches.append(mismatch)
            logger.info(f"{action.label} response: {response_text}")
            live_action_responses.append({"label": action.label, "response": response_text})
            await asyncio.sleep(args.turn_pause_secs)

    finally:
        await asyncio.sleep(args.close_delay_secs)
        await pc.close()

    await asyncio.sleep(args.log_flush_secs)

    bot_text = _read_new_text(args.bot_log, bot_log_offset)
    vllm_text = _read_new_text(args.vllm_log, vllm_log_offset)
    full_bot_log = args.bot_log.read_text(encoding="utf-8", errors="replace")
    full_vllm_log = args.vllm_log.read_text(encoding="utf-8", errors="replace")
    conversation_id = _extract_latest_conversation_id(full_bot_log)
    completed_responses = _extract_completed_responses(bot_text)
    response_source = "bot_log"
    if len(completed_responses) != len(actions):
        live_completed_responses = [
            item["response"] for item in live_action_responses
        ]
        if len(live_completed_responses) != len(actions):
            raise AssertionError(
                "Expected "
                f"{len(actions)} completed responses, found "
                f"{len(completed_responses)} in bot log and "
                f"{len(live_completed_responses)} live responses: "
                f"bot_log={completed_responses!r} "
                f"live={live_completed_responses!r}"
            )
        logger.warning(
            "Bot log response scrape found {}/{} completed responses; "
            "falling back to live RTVI responses.",
            len(completed_responses),
            len(actions),
        )
        completed_responses = live_completed_responses
        response_source = "live_rtvi"
    for action, response_text in zip(actions, completed_responses, strict=True):
        missing = _missing_expected_response_substrings(action, response_text)
        if not missing:
            continue
        mismatch = {
            "label": action.label,
            "response": response_text,
            "missing_expected_substrings": missing,
            "source": response_source,
        }
        if not args.allow_response_mismatches:
            raise AssertionError(
                f"{action.label} response missing expected substring(s) "
                f"{missing}: {response_text!r}"
            )
        if mismatch not in response_mismatches:
            logger.warning(
                "{} post-run response mismatch {}: {!r}",
                action.label,
                missing,
                response_text,
            )
            response_mismatches.append(mismatch)
    bot_analysis_text, bot_analysis_source = _select_bot_analysis_text(
        bot_text=bot_text,
        full_bot_log=full_bot_log,
        conversation_id=conversation_id,
    )
    vllm_analysis_text, vllm_analysis_source = _select_vllm_analysis_text(
        vllm_text=vllm_text,
        full_vllm_log=full_vllm_log,
        conversation_id=conversation_id,
    )
    if bot_analysis_source != "slice" or vllm_analysis_source != "slice":
        logger.warning(
            "Using log-analysis fallback bot={} vllm={} for conversation_id={}",
            bot_analysis_source,
            vllm_analysis_source,
            conversation_id,
        )
    stats = _validate_logs(
        bot_text=bot_analysis_text,
        vllm_text=vllm_analysis_text,
        conversation_id=conversation_id,
        expect_cache_attach=args.expect_cache_attach,
    )

    summary = {
        "conversation_id": conversation_id,
        "turns": len(actions),
        "expect_cache_attach": args.expect_cache_attach,
        "rtvi_message_types": received_message_types,
        "live_action_responses": live_action_responses,
        "response_source": response_source,
        "log_analysis_source": {
            "bot": bot_analysis_source,
            "vllm": vllm_analysis_source,
        },
        "completed_responses": [
            {"label": action.label, "response": response_text}
            for action, response_text in zip(actions, completed_responses, strict=True)
        ],
        "response_mismatches": response_mismatches,
        "stats": stats,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer-url", default="http://127.0.0.1:7860/api/offer")
    parser.add_argument("--bot-log", type=Path, default=DEFAULT_BOT_LOG)
    parser.add_argument("--vllm-log", type=Path, default=DEFAULT_VLLM_LOG)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--turn-timeout", type=float, default=90.0)
    parser.add_argument("--audio-send-timeout", type=float, default=30.0)
    parser.add_argument("--turn-gap-secs", type=float, default=2.0)
    parser.add_argument("--turn-pause-secs", type=float, default=0.5)
    parser.add_argument("--close-delay-secs", type=float, default=1.0)
    parser.add_argument("--log-flush-secs", type=float, default=2.0)
    parser.add_argument(
        "--expect-cache-attach",
        choices=("always", "never"),
        default="always",
    )
    parser.add_argument(
        "--allow-response-mismatches",
        action="store_true",
        help=(
            "Record semantic response mismatches in the summary instead of "
            "failing the run immediately. Infrastructure failures still fail."
        ),
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optionally run only the first N default turns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
