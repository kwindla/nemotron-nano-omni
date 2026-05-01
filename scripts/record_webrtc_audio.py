#!/usr/bin/env python3
"""Record remote audio from the local SmallWebRTC bot."""

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
from av.audio.resampler import AudioResampler
from loguru import logger


class WavThenSilenceTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(
        self,
        wav_path: Path,
        *,
        sample_rate: int = 48000,
        frame_ms: int = 20,
        silence_secs: float = 8.0,
    ):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = int(sample_rate * frame_ms / 1000)
        self._frame_duration = frame_ms / 1000
        self._pcm = self._load_wav(wav_path, sample_rate)
        self._pos = 0
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
        if self._pos < len(self._pcm):
            chunk = self._pcm[self._pos : self._pos + bytes_per_frame]
            self._pos += len(chunk)
            if len(chunk) < bytes_per_frame:
                chunk += b"\x00" * (bytes_per_frame - len(chunk))
            if self._pos >= len(self._pcm):
                logger.info("Finished sending WAV turn; sending silence")
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
    output = bytearray()
    resampler = AudioResampler(format="s16", layout="mono", rate=args.output_sample_rate)
    got_audio = asyncio.Event()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        logger.info(f"Receiving remote {track.kind} track")

        async def receive():
            while True:
                try:
                    frame = await track.recv()
                except Exception as e:
                    logger.info(f"Remote {track.kind} track ended: {e}")
                    return
                if track.kind != "audio":
                    continue
                for out_frame in resampler.resample(frame):
                    output.extend(bytes(out_frame.planes[0]))
                    got_audio.set()

        asyncio.create_task(receive())

    pc.addTrack(
        WavThenSilenceTrack(
            args.audio,
            sample_rate=args.input_sample_rate,
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
        await asyncio.wait_for(got_audio.wait(), timeout=args.first_audio_timeout)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for first remote audio frame")

    await asyncio.sleep(args.record_secs)
    logger.info(f"Captured {len(output)} bytes of remote audio")
    await pc.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(args.output_sample_rate)
        wav.writeframes(bytes(output))
    logger.info(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer-url", default="http://127.0.0.1:7860/api/offer")
    parser.add_argument("--audio", type=Path, default=Path("media/cartesia-unicorn.wav"))
    parser.add_argument("--output", type=Path, default=Path("media/webrtc-bot-output.wav"))
    parser.add_argument("--input-sample-rate", type=int, default=48000)
    parser.add_argument("--output-sample-rate", type=int, default=48000)
    parser.add_argument("--silence-secs", type=float, default=8.0)
    parser.add_argument("--record-secs", type=float, default=28.0)
    parser.add_argument("--first-audio-timeout", type=float, default=20.0)
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
