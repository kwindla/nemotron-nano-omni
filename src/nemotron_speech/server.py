"""WebSocket ASR server for local Nemotron Speech streaming inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from aiohttp import WSMsgType, web
from loguru import logger

DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


@dataclass
class ASRSession:
    """Per-connection streaming ASR state."""

    id: str
    websocket: web.WebSocketResponse
    accumulated_audio: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    emitted_frames: int = 0
    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None
    previous_hypotheses: Any = None
    pred_out_stream: Any = None
    current_text: str = ""
    last_emitted_text: str = ""


class ASRServer:
    """Small HTTP/WebSocket server for NeMo cache-aware streaming ASR.

    Wire protocol:
    - binary WebSocket messages are 16 kHz mono PCM16 audio chunks
    - text message {"type": "reset", "finalize": true} finalizes a turn
    - server transcript messages use {"type": "transcript", "text": ..., "is_final": ...}
    """

    def __init__(
        self,
        *,
        model: str,
        host: str,
        port: int,
        right_context: int,
        device: str | None,
        warmup: bool,
    ) -> None:
        self.model_name_or_path = model
        self.host = host
        self.port = port
        self.right_context = right_context
        requested_device = (device or os.getenv("NEMOTRON_SPEECH_DEVICE", "cuda")).strip()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested for ASR but unavailable; falling back to CPU")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.warmup = warmup

        self.model: Any = None
        self.model_loaded = False
        self.sample_rate = 16000
        self.hop_samples = 160
        self.shift_frames = 16
        self.pre_encode_cache_size = 0
        self.drop_extra = 0
        self.final_padding_frames = 32

        self.inference_lock = asyncio.Lock()
        self.sessions: dict[str, ASRSession] = {}

    def load_model(self) -> None:
        """Load and configure the NeMo ASR model."""
        import nemo.collections.asr as nemo_asr

        torch.set_float32_matmul_precision("high")
        model_ref_is_path = self.model_name_or_path.endswith(".nemo") or os.path.exists(
            self.model_name_or_path
        )
        map_location = torch.device(self.device)

        if model_ref_is_path:
            logger.info(f"Loading ASR model from {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                self.model_name_or_path,
                map_location=map_location,
            )
        else:
            logger.info(f"Loading ASR model from Hugging Face: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name_or_path,
                map_location=map_location,
            )

        self.model = self.model.to(self.device)
        self._configure_streaming()
        self._configure_decoding()
        self.model.eval()

        if hasattr(self.model, "preprocessor") and hasattr(
            self.model.preprocessor, "featurizer"
        ):
            self.model.preprocessor.featurizer.dither = 0.0

        self._derive_streaming_sizes()
        if self.warmup:
            self._warmup()
        self.model_loaded = True

    def _configure_streaming(self) -> None:
        if not hasattr(self.model.encoder, "set_default_att_context_size"):
            raise RuntimeError("ASR model encoder does not support streaming attention context")
        att_context_size = [70, self.right_context]
        logger.info(
            "Setting ASR attention context to "
            f"{att_context_size} ({RIGHT_CONTEXT_OPTIONS.get(self.right_context, 'custom')})"
        )
        self.model.encoder.set_default_att_context_size(att_context_size=att_context_size)

    def _configure_decoding(self) -> None:
        """Use greedy decoding and avoid CUDA graph decoder paths on Blackwell."""
        from omegaconf import OmegaConf, open_dict

        decoding_cfg = getattr(getattr(self.model, "cfg", None), "decoding", None)
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.create(
                {
                    "strategy": "greedy",
                    "greedy": {
                        "max_symbols": 10,
                        "loop_labels": False,
                        "use_cuda_graph_decoder": False,
                    },
                }
            )
        else:
            decoding_cfg = OmegaConf.create(OmegaConf.to_container(decoding_cfg))
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"
                decoding_cfg.compute_timestamps = False
                if "greedy" not in decoding_cfg or decoding_cfg.greedy is None:
                    decoding_cfg.greedy = {}
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.greedy.loop_labels = False
                decoding_cfg.greedy.use_cuda_graph_decoder = False
                if "fused_batch_size" in decoding_cfg:
                    decoding_cfg.fused_batch_size = -1

        logger.info("Configuring ASR greedy decoding for Blackwell compatibility")
        if hasattr(self.model, "cur_decoder"):
            decoder_type = getattr(self.model, "cur_decoder", None)
            self.model.change_decoding_strategy(decoding_cfg, decoder_type=decoder_type)
        else:
            self.model.change_decoding_strategy(decoding_cfg)

    def _derive_streaming_sizes(self) -> None:
        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get("window_stride", 0.01)
        self.hop_samples = int(hop_length_sec * self.sample_rate)

        streaming_cfg = self.model.encoder.streaming_cfg
        shift_size = streaming_cfg.shift_size
        self.shift_frames = shift_size[1] if isinstance(shift_size, list) else shift_size

        pre_cache = streaming_cfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
        self.drop_extra = streaming_cfg.drop_extra_pre_encoded
        self.final_padding_frames = (self.right_context + 1) * self.shift_frames

        logger.info(
            "ASR streaming config: "
            f"shift={self.shift_frames} frames, "
            f"pre_encode_cache={self.pre_encode_cache_size} frames, "
            f"final_padding={self.final_padding_frames} frames"
        )

    def _warmup(self) -> None:
        logger.info("Running ASR warmup with streaming API")
        start = time.perf_counter()
        warmup_samples = self.sample_rate + self.final_padding_frames * self.hop_samples
        warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
        session = ASRSession(id="warmup", websocket=None)  # type: ignore[arg-type]
        self._init_session(session)
        session.accumulated_audio = warmup_audio
        self._process_final_chunk(session)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"ASR warmup complete in {elapsed_ms:.0f}ms")

    def _initial_cache(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            return self.model.encoder.get_initial_cache_state(
                batch_size=1,
                device=self.device,
            )
        except TypeError:
            cache = self.model.encoder.get_initial_cache_state(batch_size=1)
            return tuple(t.to(self.device) for t in cache)  # type: ignore[return-value]

    def _init_session(self, session: ASRSession) -> None:
        (
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
        ) = self._initial_cache()
        session.accumulated_audio = np.array([], dtype=np.float32)
        session.emitted_frames = 0
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        session_id = uuid.uuid4().hex[:8]
        session = ASRSession(id=session_id, websocket=ws)
        self.sessions[session_id] = session
        logger.info(f"ASR client {session_id} connected")

        try:
            async with self.inference_lock:
                await asyncio.to_thread(self._init_session, session)
            await ws.send_str(json.dumps({"type": "ready"}))

            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._handle_control_message(session, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"ASR client {session_id} WebSocket error: {ws.exception()}")
                    break
        except Exception as exc:
            logger.exception(f"ASR client {session_id} failed: {exc}")
            if not ws.closed:
                await ws.send_str(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            self.sessions.pop(session_id, None)
            logger.info(f"ASR client {session_id} disconnected")

        return ws

    async def _handle_control_message(self, session: ASRSession, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"ASR client {session.id}: invalid JSON control message")
            return

        if data.get("type") not in {"reset", "end"}:
            logger.warning(f"ASR client {session.id}: unknown control message {data!r}")
            return

        await self._reset_session(session, finalize=data.get("finalize", True))

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes) -> None:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_np])

        min_audio_for_chunk = (
            session.emitted_frames + self.shift_frames + 1
        ) * self.hop_samples
        while len(session.accumulated_audio) >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await asyncio.to_thread(self._process_chunk, session)

            if text is not None and text != session.current_text:
                session.current_text = text
                await session.websocket.send_str(
                    json.dumps({"type": "transcript", "text": text, "is_final": False})
                )

            min_audio_for_chunk = (
                session.emitted_frames + self.shift_frames + 1
            ) * self.hop_samples

    def _audio_to_mel(self, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([len(audio)], device=self.device)
        return self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

    def _process_chunk(self, session: ASRSession) -> str | None:
        try:
            with torch.inference_mode():
                mel, _mel_len = self._audio_to_mel(session.accumulated_audio)
                available_frames = mel.shape[-1] - 1
                new_frame_count = available_frames - session.emitted_frames
                if new_frame_count < self.shift_frames:
                    return session.current_text

                if session.emitted_frames == 0:
                    chunk_start = 0
                    chunk_end = self.shift_frames
                    drop_extra = 0
                else:
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    chunk_end = session.emitted_frames + self.shift_frames
                    drop_extra = self.drop_extra

                chunk_mel = mel[:, :, chunk_start:chunk_end]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
                text = self._stream_step(
                    session,
                    chunk_mel=chunk_mel,
                    chunk_len=chunk_len,
                    keep_all_outputs=False,
                    drop_extra=drop_extra,
                )
                session.emitted_frames += self.shift_frames
                return text
        except Exception as exc:
            logger.exception(f"ASR session {session.id} chunk processing failed: {exc}")
            return None

    def _process_final_chunk(self, session: ASRSession) -> str | None:
        try:
            if len(session.accumulated_audio) == 0:
                return session.current_text

            with torch.inference_mode():
                mel, _mel_len = self._audio_to_mel(session.accumulated_audio)
                total_mel_frames = mel.shape[-1]
                remaining_frames = total_mel_frames - session.emitted_frames
                if remaining_frames <= 0:
                    return session.current_text

                if session.emitted_frames == 0:
                    chunk_start = 0
                    drop_extra = 0
                else:
                    chunk_start = session.emitted_frames - self.pre_encode_cache_size
                    drop_extra = self.drop_extra

                chunk_mel = mel[:, :, chunk_start:]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
                return self._stream_step(
                    session,
                    chunk_mel=chunk_mel,
                    chunk_len=chunk_len,
                    keep_all_outputs=True,
                    drop_extra=drop_extra,
                )
        except Exception as exc:
            logger.exception(f"ASR session {session.id} final processing failed: {exc}")
            return None

    def _stream_step(
        self,
        session: ASRSession,
        *,
        chunk_mel: torch.Tensor,
        chunk_len: torch.Tensor,
        keep_all_outputs: bool,
        drop_extra: int,
    ) -> str:
        (
            session.pred_out_stream,
            transcribed_texts,
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.previous_hypotheses,
        ) = self.model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.pred_out_stream,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )
        return self._extract_text(transcribed_texts) or session.current_text

    @staticmethod
    def _extract_text(transcribed_texts: Any) -> str | None:
        if not transcribed_texts:
            return None
        hyp = transcribed_texts[0]
        if hasattr(hyp, "text"):
            return hyp.text
        if isinstance(hyp, str):
            return hyp
        return str(hyp)

    async def _reset_session(self, session: ASRSession, *, finalize: bool) -> None:
        logger.debug(
            f"ASR session {session.id} reset finalize={finalize} "
            f"audio_samples={len(session.accumulated_audio)}"
        )
        if not finalize:
            await session.websocket.send_str(
                json.dumps(
                    {
                        "type": "transcript",
                        "text": session.current_text,
                        "is_final": True,
                        "finalize": False,
                    }
                )
            )
            return

        if len(session.accumulated_audio) > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate(
                [session.accumulated_audio, silence_padding]
            )

        final_text = session.current_text
        async with self.inference_lock:
            text = await asyncio.to_thread(self._process_final_chunk, session)
        if text is not None:
            final_text = text

        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text) :].lstrip()
        else:
            delta_text = final_text
        session.last_emitted_text = final_text

        await session.websocket.send_str(
            json.dumps(
                {
                    "type": "transcript",
                    "text": delta_text,
                    "is_final": True,
                    "finalize": True,
                }
            )
        )
        session.last_emitted_text = ""
        self._init_session(session)

    async def health_handler(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "healthy" if self.model_loaded else "loading",
                "model_loaded": self.model_loaded,
                "model": self.model_name_or_path,
                "device": str(self.device),
                "right_context": self.right_context,
                "sessions": len(self.sessions),
            }
        )

    async def start(self) -> None:
        self.load_model()
        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/", self.websocket_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"ASR WebSocket server listening on ws://{self.host}:{self.port}")
        logger.info(f"ASR health check listening on http://{self.host}:{self.port}/health")
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nemotron Speech streaming ASR server")
    parser.add_argument("--host", default=os.getenv("NEMOTRON_SPEECH_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("NEMOTRON_SPEECH_PORT", "8080")),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("NEMOTRON_SPEECH_MODEL", DEFAULT_MODEL),
        help="Hugging Face model id or local .nemo path",
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=int(os.getenv("NEMOTRON_SPEECH_RIGHT_CONTEXT", "1")),
        choices=sorted(RIGHT_CONTEXT_OPTIONS),
    )
    parser.add_argument("--device", default=os.getenv("NEMOTRON_SPEECH_DEVICE", "cuda"))
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip startup streaming warmup.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    server = ASRServer(
        model=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
        device=args.device,
        warmup=not args.no_warmup,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()

