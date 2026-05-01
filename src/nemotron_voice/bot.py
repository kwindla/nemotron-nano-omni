#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SmallWebRTC bot for local Nemotron Omni audio-input testing."""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from nemotron_voice.services.kyutai.tts import PocketTTSService
from nemotron_voice.services.nvidia.nemotron_omni import (
    DEFAULT_VOICE_SYSTEM_INSTRUCTION,
    NemotronOmniAudioLLMService,
)
from nemotron_voice.services.nvidia.nemotron_speech import NemotronSpeechWebSocketSTTService
from nemotron_voice.services.nvidia.nemotron_tts import NemotronMagpieWebSocketTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

_FILE_LOGGER_ID: int | None = None
_SYSTEM_INSTRUCTION_ENV = "NEMOTRON_OMNI_SYSTEM_INSTRUCTION"


def _load_env():
    for env_path in (
        Path.cwd() / ".env",
        Path.home() / ".env" / "pipecat",
        Path("/home/khkramer/src/pipecat/.env"),
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    load_dotenv(override=False)


def _configure_logging():
    # The Pipecat runner configures stderr logging when main() starts. We do a
    # light pre-run setup here, then add the file sink from run_bot() after the
    # runner has finished its own logger setup.
    log_path = Path(os.getenv("NEMOTRON_OMNI_LOG", "nemotron-omni-audio-bot.log"))
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("NEMOTRON_OMNI_STDERR_LEVEL", "INFO"))
    logger.info(f"Debug log: {log_path.resolve()}")


_load_env()
_configure_logging()


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def _ensure_file_logging():
    global _FILE_LOGGER_ID
    if _FILE_LOGGER_ID is not None:
        return

    log_path = Path(os.getenv("NEMOTRON_OMNI_LOG", "nemotron-omni-audio-bot.log"))
    _FILE_LOGGER_ID = logger.add(log_path, level="DEBUG", backtrace=True, diagnose=False)
    logger.info(f"Debug log: {log_path.resolve()}")


def _build_nemotron_speech_stt(*, audio_passthrough: bool = False) -> NemotronSpeechWebSocketSTTService:
    url = os.getenv("NEMOTRON_SPEECH_STT_URL", "ws://127.0.0.1:8080")
    logger.info(f"Using local Nemotron Speech WebSocket STT at {url}")

    return NemotronSpeechWebSocketSTTService(
        url=url,
        sample_rate=16000,
        language=Language.EN_US,
        audio_passthrough=audio_passthrough,
    )


def _build_nemotron_magpie_tts() -> NemotronMagpieWebSocketTTSService:
    url = os.getenv("NEMOTRON_MAGPIE_TTS_URL", os.getenv("NVIDIA_TTS_URL", "http://127.0.0.1:8001"))
    voice = os.getenv("NEMOTRON_MAGPIE_TTS_VOICE", "aria")
    language = os.getenv("NEMOTRON_MAGPIE_TTS_LANGUAGE", "en")
    logger.info(f"Using local NVIDIA Magpie WebSocket TTS at {url}")

    return NemotronMagpieWebSocketTTSService(
        server_url=url,
        voice=voice,
        language=language,
        params=NemotronMagpieWebSocketTTSService.InputParams(
            streaming_preset=os.getenv("NEMOTRON_MAGPIE_TTS_STREAMING_PRESET", "conservative"),
            use_adaptive_mode=os.getenv("NEMOTRON_MAGPIE_TTS_ADAPTIVE", "1") != "0",
            sentence_pause_ms=int(os.getenv("NEMOTRON_MAGPIE_TTS_SENTENCE_PAUSE_MS", "250")),
        ),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )


def _build_pocket_tts() -> PocketTTSService:
    url = os.getenv("POCKET_TTS_URL", os.getenv("KYUTAI_POCKET_TTS_URL", "http://127.0.0.1:8001"))
    voice = os.getenv("POCKET_TTS_VOICE", "alba")
    logger.info(f"Using local Kyutai Pocket TTS at {url} with voice {voice}")

    return PocketTTSService(
        base_url=url,
        voice=voice,
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )


def _build_tts():
    provider = os.getenv("NEMOTRON_TTS_PROVIDER", "pocket").strip().lower()
    if provider in {"pocket", "kyutai", "pocket-tts"}:
        return _build_pocket_tts()
    if provider in {"magpie", "nvidia", "nemo"}:
        return _build_nemotron_magpie_tts()
    raise ValueError(f"Unsupported NEMOTRON_TTS_PROVIDER={provider!r}")


class SmartTurnRTVIObserver(BaseObserver):
    """Expose Smart Turn predictions as custom RTVI server messages."""

    def __init__(self, *, rtvi: RTVIProcessor):
        super().__init__()
        self._rtvi = rtvi
        self._seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if frame.id in self._seen_frame_ids or not isinstance(frame, MetricsFrame):
            return

        turn_metrics = [metric for metric in frame.data if isinstance(metric, TurnMetricsData)]
        if not turn_metrics:
            return

        self._seen_frame_ids.add(frame.id)
        for metric in turn_metrics:
            payload = {
                "type": "smart-turn",
                "state": "complete" if metric.is_complete else "incomplete",
                "complete": metric.is_complete,
                "probability": metric.probability,
                "e2e_processing_time_ms": metric.e2e_processing_time_ms,
                "processor": metric.processor,
            }
            await self._rtvi.send_server_message(payload)
            logger.debug(f"Sent Smart Turn RTVI server message: {payload}")


class AudioOnlySmartTurnStopStrategy(BaseUserTurnStopStrategy):
    """Stop a user turn from Smart Turn's audio classification only."""

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._vad_user_speaking = False

    async def reset(self):
        await super().reset()
        self._vad_user_speaking = False

    async def cleanup(self):
        await super().cleanup()
        await self._turn_analyzer.cleanup()

    async def process_frame(self, frame: Frame):
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._turn_analyzer.update_vad_start_secs(frame.start_secs)
            self._vad_user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_user_speaking = False
            await self._analyze_end_of_turn()
        elif isinstance(frame, InputAudioRawFrame):
            self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

    async def _start(self, frame: StartFrame):
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(
            SpeechControlParamsFrame, turn_params=self._turn_analyzer.params
        )

    async def _analyze_end_of_turn(self):
        state, result = await self._turn_analyzer.analyze_end_of_turn()
        if result:
            await self.push_frame(MetricsFrame(data=[result]))

        # Only trigger from the model classification result. BaseSmartTurn can
        # also return COMPLETE from a silence timeout without metrics; leave
        # that to the user-turn controller's normal timeout fallback.
        if state is EndOfTurnState.COMPLETE and result is not None:
            is_complete = getattr(result, "is_complete", True)
            if is_complete:
                await self.trigger_user_turn_stopped()


class UserAudioContextCollector(FrameProcessor):
    """Collect one user audio turn and append it to the shared LLM context."""

    def __init__(
        self,
        *,
        context: LLMContext,
        user_aggregator: LLMUserAggregator,
        audio_context_text: str,
        push_context_on_finish: bool = True,
        pre_speech_buffer_secs: float = 0.2,
    ):
        super().__init__()
        self._context = context
        self._user_aggregator = user_aggregator
        self._audio_context_text = audio_context_text
        self._push_context_on_finish = push_context_on_finish
        self._pre_speech_buffer_secs = pre_speech_buffer_secs
        self._audio_frames: list[InputAudioRawFrame] = []
        self._user_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._finish_user_turn()
        elif isinstance(frame, InputAudioRawFrame):
            self._collect_audio_frame(frame)
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._user_speaking:
            await self._finish_user_turn()

        await self.push_frame(frame, direction)

    def _collect_audio_frame(self, frame: InputAudioRawFrame):
        if not frame.audio:
            return

        self._audio_frames.append(frame)
        if self._user_speaking:
            return

        duration = frame.num_frames / frame.sample_rate if frame.sample_rate else 0
        buffered_duration = duration * len(self._audio_frames)
        while self._audio_frames and buffered_duration > self._pre_speech_buffer_secs:
            self._audio_frames.pop(0)
            buffered_duration -= duration

    async def _finish_user_turn(self):
        if not self._audio_frames:
            self._user_speaking = False
            return

        audio_frames = list(self._audio_frames)
        self._audio_frames.clear()
        self._user_speaking = False

        await self._context.add_audio_frames_message(
            audio_frames=audio_frames,
            text=self._audio_context_text,
        )
        logger.debug(
            "Added user audio turn to LLM context "
            f"({len(audio_frames)} frames, {len(self._context.get_messages())} messages)"
        )
        if self._push_context_on_finish:
            await self._user_aggregator.push_context_frame()


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    _ensure_file_logging()
    logger.info("Starting Nemotron Omni audio bot")
    conversation_id = os.getenv("NEMOTRON_OMNI_CONVERSATION_ID") or (
        f"pipecat-{uuid.uuid4().hex}"
    )
    logger.info(f"Using Nemotron Omni conversation_id={conversation_id}")

    rtvi = RTVIProcessor()

    async def send_bash_tool_event(payload: dict):
        await rtvi.send_server_message(payload)

    llm = NemotronOmniAudioLLMService(
        base_url=os.getenv("NEMOTRON_OMNI_BASE_URL", "http://127.0.0.1:8000/v1"),
        conversation_id=conversation_id,
        suffix_only_conversation=(
            os.getenv("NEMOTRON_OMNI_SUFFIX_ONLY_CONVERSATION", "1") != "0"
        ),
        enable_bash_tool=os.getenv("NEMOTRON_OMNI_ENABLE_BASH_TOOL", "1") != "0",
        bash_tool_cwd=os.getenv("NEMOTRON_OMNI_BASH_TOOL_CWD", str(Path.cwd())),
        bash_tool_timeout_secs=float(os.getenv("NEMOTRON_OMNI_BASH_TOOL_TIMEOUT_SECS", "20")),
        bash_tool_max_output_chars=int(
            os.getenv("NEMOTRON_OMNI_BASH_TOOL_MAX_OUTPUT_CHARS", "12000")
        ),
        bash_tool_event_sender=send_bash_tool_event,
        settings=NemotronOmniAudioLLMService.Settings(
            system_instruction=os.getenv(
                _SYSTEM_INSTRUCTION_ENV, DEFAULT_VOICE_SYSTEM_INSTRUCTION
            ),
            max_tokens=int(os.getenv("NEMOTRON_OMNI_MAX_TOKENS", "256")),
            temperature=0.0,
            top_k=1,
            audio_prompt="Listen to the audio and respond to the spoken instruction.",
            chat_template_kwargs={"enable_thinking": False},
        ),
    )

    tts = _build_tts()
    stt = _build_nemotron_speech_stt(audio_passthrough=False)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    AudioOnlySmartTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
        ),
    )
    audio_collector = UserAudioContextCollector(
        context=context,
        user_aggregator=user_aggregator,
        audio_context_text=os.getenv(
            "NEMOTRON_OMNI_AUDIO_CONTEXT_TEXT",
            "User audio follows. Listen to it and respond to the user's latest request.",
        ),
        push_context_on_finish=True,
    )
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            ParallelPipeline(
                [
                    audio_collector,
                    llm,
                    tts,
                    transport.output(),
                    assistant_aggregator,
                ],
                [
                    stt,
                ],
            ),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        rtvi_processor=rtvi,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[
            DebugLogObserver(
                frame_types={
                    LLMFullResponseStartFrame: None,
                    LLMFullResponseEndFrame: None,
                    LLMTextFrame: None,
                    LLMMessagesAppendFrame: None,
                    TranscriptionFrame: None,
                    InterimTranscriptionFrame: None,
                    TTSTextFrame: None,
                    TTSAudioRawFrame: (NemotronMagpieWebSocketTTSService, FrameEndpoint.SOURCE),
                    UserStoppedSpeakingFrame: None,
                    VADUserStartedSpeakingFrame: None,
                    VADUserStoppedSpeakingFrame: None,
                    ErrorFrame: None,
                }
            ),
            SmartTurnRTVIObserver(rtvi=rtvi),
        ],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected; waiting for user audio")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with the Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
