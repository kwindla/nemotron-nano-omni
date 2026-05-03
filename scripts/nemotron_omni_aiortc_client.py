#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hand-coded aiortc client for the Nemotron Omni SmallWebRTC test bot."""

from __future__ import annotations

import argparse
import asyncio
import audioop
import json
import wave
from fractions import Fraction
from pathlib import Path

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


class WavThenSilenceTrack(AudioStreamTrack):
    """Audio track that sends one or more WAV turns separated by silence."""

    kind = "audio"

    def __init__(
        self,
        wav_paths: list[Path],
        *,
        sample_rate: int = 48000,
        frame_ms: int = 20,
        turn_gap_secs: float = 12.0,
        silence_secs: float = 20.0,
    ):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = int(sample_rate * frame_ms / 1000)
        self._frame_duration = frame_ms / 1000
        self._pcm_turns = [self._load_wav(path, sample_rate) for path in wav_paths]
        self._current_turn = 0
        self._pos = 0
        self._turn_gap_bytes = int(sample_rate * 2 * turn_gap_secs)
        self._gap_bytes_remaining = 0
        self._silence_bytes_remaining = int(sample_rate * 2 * silence_secs)
        self._pts = 0

    @staticmethod
    def _load_wav(path: Path, target_sample_rate: int) -> bytes:
        with wave.open(str(path), "rb") as wav:
            num_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())

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

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(self._frame_duration)

        bytes_per_frame = self._samples_per_frame * 2
        if self._gap_bytes_remaining > 0:
            chunk = b"\x00" * bytes_per_frame
            self._gap_bytes_remaining -= bytes_per_frame
        elif self._current_turn < len(self._pcm_turns) and self._pos < len(
            self._pcm_turns[self._current_turn]
        ):
            pcm = self._pcm_turns[self._current_turn]
            chunk = pcm[self._pos : self._pos + bytes_per_frame]
            self._pos += len(chunk)
            if len(chunk) < bytes_per_frame:
                chunk += b"\x00" * (bytes_per_frame - len(chunk))
            if self._pos >= len(pcm):
                logger.info(f"Finished sending WAV turn {self._current_turn + 1}")
                self._current_turn += 1
                self._pos = 0
                if self._current_turn < len(self._pcm_turns):
                    logger.info("Sending inter-turn silence to trigger VAD stop")
                    self._gap_bytes_remaining = self._turn_gap_bytes
                else:
                    logger.info("Finished all WAV turns; sending silence to trigger VAD stop")
        elif self._silence_bytes_remaining > 0:
            chunk = b"\x00" * bytes_per_frame
            self._silence_bytes_remaining -= bytes_per_frame
        else:
            chunk = b"\x00" * bytes_per_frame

        frame = AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.planes[0].update(chunk)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += self._samples_per_frame
        return frame


async def _wait_for_ice_gathering(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change():
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def run(args: argparse.Namespace):
    pc = RTCPeerConnection()
    audio_frames_received = 0

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        logger.info(f"Receiving remote {track.kind} track")

        async def receive():
            nonlocal audio_frames_received
            while True:
                try:
                    frame = await track.recv()
                except Exception as e:
                    logger.info(f"Remote {track.kind} track ended: {e}")
                    return
                if track.kind == "audio":
                    audio_frames_received += 1
                    if audio_frames_received == 1:
                        logger.info("Received first bot audio frame")

        asyncio.create_task(receive())

    pc.addTrack(
        WavThenSilenceTrack(
            [Path(path) for path in args.audio],
            sample_rate=args.sample_rate,
            turn_gap_secs=args.turn_gap_secs,
            silence_secs=args.silence_secs,
        )
    )

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _wait_for_ice_gathering(pc)

    payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    logger.info(f"Posting offer to {args.offer_url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(args.offer_url, json=payload) as response:
            response.raise_for_status()
            answer = await response.json()

    logger.debug(json.dumps({"answer": answer["type"], "pc_id": answer.get("pc_id")}, indent=2))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    try:
        await asyncio.sleep(args.run_secs)
    finally:
        logger.info(f"Received {audio_frames_received} bot audio frames")
        await pc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offer-url",
        default="http://127.0.0.1:7860/api/offer",
        help="SmallWebRTC offer endpoint.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        help="WAV file to stream as one user turn. Repeat for multiple turns.",
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--turn-gap-secs", type=float, default=12.0)
    parser.add_argument("--silence-secs", type=float, default=5.0)
    parser.add_argument("--run-secs", type=float, default=35.0)
    args = parser.parse_args()
    if not args.audio:
        args.audio = [str(ROOT / "media" / "cartesia-unicorn.wav")]

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
