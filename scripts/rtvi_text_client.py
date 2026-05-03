#!/usr/bin/env python3
"""Drive SmallWebRTC text turns through the RTVI data channel."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from loguru import logger


async def _wait_for_ice_gathering(pc: RTCPeerConnection):
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change():
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


def _rtvi_message(msg_type: str, data: dict | None = None) -> str:
    payload = {
        "label": "rtvi-ai",
        "type": msg_type,
        "id": uuid.uuid4().hex,
    }
    if data is not None:
        payload["data"] = data
    return json.dumps(payload)


async def run(args: argparse.Namespace):
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("rtvi-ai")
    channel_open = asyncio.Event()
    bot_ready = asyncio.Event()
    turn_done = asyncio.Event()
    llm_started = asyncio.Event()
    received_types: list[str] = []
    llm_text_chunks: list[str] = []
    tts_text_chunks: list[str] = []
    current_turn = 0

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        logger.info(f"Receiving remote {track.kind} track")

        async def receive():
            got_first_frame = False
            while True:
                try:
                    await track.recv()
                except Exception as exc:
                    logger.info(f"Remote {track.kind} track ended: {exc}")
                    return
                if not got_first_frame:
                    got_first_frame = True
                    logger.info(f"Received first remote {track.kind} frame")

        asyncio.create_task(receive())

    @channel.on("open")
    def on_open():
        logger.info("RTVI data channel open")
        channel.send(
            _rtvi_message(
                "client-ready",
                {
                    "version": "1.2.0",
                    "about": {
                        "library": "custom-aiortc",
                        "library_version": "0",
                        "platform": "python",
                    },
                },
            )
        )
        channel_open.set()

    @channel.on("message")
    def on_message(message):
        if not isinstance(message, str):
            logger.info(f"RTVI binary message ignored ({len(message)} bytes)")
            return

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.info(f"RTVI non-JSON message: {message}")
            return

        msg_type = payload.get("type", "<missing>")
        received_types.append(msg_type)

        if msg_type == "bot-ready":
            logger.info("Received bot-ready")
            bot_ready.set()
            return

        if msg_type == "bot-llm-started":
            logger.info(f"Turn {current_turn}: bot-llm-started")
            llm_started.set()
            return

        if msg_type == "bot-llm-text":
            text = ((payload.get("data") or {}).get("text")) or ""
            if text:
                llm_text_chunks.append(text)
            return

        if msg_type == "bot-tts-text":
            text = ((payload.get("data") or {}).get("text")) or ""
            if text:
                tts_text_chunks.append(text)
            return

        if msg_type == "bot-llm-stopped":
            logger.info(f"Turn {current_turn}: bot-llm-stopped")
            if not args.audio_response:
                turn_done.set()
            return

        if msg_type == "bot-tts-stopped":
            logger.info(f"Turn {current_turn}: bot-tts-stopped")
            if args.audio_response:
                turn_done.set()
            return

        if msg_type == "error-response":
            logger.error(f"RTVI error-response: {payload}")
            turn_done.set()
            return

        if msg_type == "error":
            logger.error(f"RTVI error: {payload}")
            turn_done.set()
            return

        logger.info(f"RTVI message: {msg_type}")

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

    logger.info(f"Received answer for pc_id={answer.get('pc_id')}")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    try:
        await asyncio.wait_for(channel_open.wait(), timeout=args.connect_timeout)
        await asyncio.wait_for(bot_ready.wait(), timeout=args.connect_timeout)

        for turn_num, text in enumerate(args.text, start=1):
            current_turn = turn_num
            turn_done.clear()
            llm_started.clear()
            logger.info(f"Sending text turn {turn_num}: {text}")
            channel.send(
                _rtvi_message(
                    "send-text",
                    {
                        "content": text,
                        "options": {
                            "run_immediately": True,
                            "audio_response": args.audio_response,
                        },
                    },
                )
            )
            await asyncio.wait_for(llm_started.wait(), timeout=args.turn_timeout)
            await asyncio.wait_for(turn_done.wait(), timeout=args.turn_timeout)
            await asyncio.sleep(args.turn_pause_secs)
    finally:
        await asyncio.sleep(args.close_delay_secs)
        await pc.close()

    logger.info(f"Received RTVI message types: {received_types}")
    if llm_text_chunks:
        logger.info(f"Aggregated bot-llm-text: {''.join(llm_text_chunks)}")
    if tts_text_chunks:
        logger.info(f"Aggregated bot-tts-text: {''.join(tts_text_chunks)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer-url", default="http://127.0.0.1:7860/api/offer")
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--audio-response", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument("--turn-timeout", type=float, default=90.0)
    parser.add_argument("--turn-pause-secs", type=float, default=1.0)
    parser.add_argument("--close-delay-secs", type=float, default=1.0)
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
