#!/usr/bin/env python3
"""Smoke test for the local ASR and Smart Turn bot wiring.

The default mode starts a tiny local ASR stub that implements the same WebSocket
protocol as ``nemotron_speech.server``. Use ``--asr-mode real`` to test against a
running real ASR server, usually ``ws://127.0.0.1:8080``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-pipecat" / "bin" / "python"
BOT = ROOT / "src/nemotron_voice/bot.py"
CLIENT = ROOT / "scripts/nemotron_omni_aiortc_client.py"
DEFAULT_AUDIO = ROOT / "media/cartesia-unicorn.wav"


async def wait_http_ok(url: str, timeout_secs: float):
    deadline = time.monotonic() + timeout_secs
    async with aiohttp.ClientSession() as session:
        last_error = None
        while time.monotonic() < deadline:
            try:
                async with session.get(url) as response:
                    if response.status < 500:
                        return
                    last_error = f"status {response.status}"
            except Exception as e:
                last_error = str(e)
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


async def wait_log_patterns(log_path: Path, patterns: list[str], timeout_secs: float):
    deadline = time.monotonic() + timeout_secs
    remaining = set(patterns)
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            remaining = {pattern for pattern in remaining if pattern not in text}
            if not remaining:
                return
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for log patterns: {sorted(remaining)}")


async def start_asr_stub(host: str, port: int, transcript: str):
    async def websocket_handler(request: web.Request):
        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "ready"}))
        sent_interim = False

        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                if not sent_interim:
                    sent_interim = True
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "transcript",
                                "text": transcript[: max(1, len(transcript) // 2)],
                                "is_final": False,
                            }
                        )
                    )
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") in {"reset", "end"}:
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "transcript",
                                "text": transcript,
                                "is_final": True,
                                "finalize": data.get("finalize", True),
                            }
                        )
                    )

        return ws

    async def health_handler(request: web.Request):
        return web.json_response({"status": "healthy", "model_loaded": True, "stub": True})

    app = web.Application()
    app.router.add_get("/", websocket_handler)
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner


async def run_subprocess(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
):
    return await asyncio.create_subprocess_exec(
        *args,
        cwd=ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.STDOUT if capture_output else asyncio.subprocess.DEVNULL,
    )


async def terminate(proc: asyncio.subprocess.Process):
    if proc.returncode is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def main_async(args: argparse.Namespace):
    if not PYTHON.exists():
        raise RuntimeError(f"Missing Pipecat Python: {PYTHON}")
    if not args.audio.exists():
        raise RuntimeError(f"Missing test audio: {args.audio}")

    asr_runner = None
    bot_proc = None
    client_proc = None
    log_path = args.log.resolve()
    log_path.unlink(missing_ok=True)

    try:
        asr_url = args.asr_url
        if args.asr_mode == "stub":
            asr_runner = await start_asr_stub(args.asr_host, args.asr_port, args.stub_transcript)
            asr_url = f"ws://{args.asr_host}:{args.asr_port}"
            await wait_http_ok(f"http://{args.asr_host}:{args.asr_port}/health", 5)
        else:
            await wait_http_ok(args.asr_health_url, args.service_timeout_secs)

        if not args.skip_vllm_health:
            await wait_http_ok(args.vllm_health_url, args.service_timeout_secs)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT / "src")
        env.update(
            {
                "NEMOTRON_SPEECH_STT_URL": asr_url,
                "NEMOTRON_OMNI_LOG": str(log_path),
                "NEMOTRON_OMNI_MAX_TOKENS": str(args.max_tokens),
                "NEMOTRON_TTS_PROVIDER": args.tts_provider,
            }
        )

        bot_proc = await run_subprocess(
            [
                str(PYTHON),
                str(BOT),
                "-t",
                "webrtc",
                "--host",
                args.bot_host,
                "--port",
                str(args.bot_port),
            ],
            env=env,
            capture_output=False,
        )
        await wait_http_ok(f"http://127.0.0.1:{args.bot_port}/client", args.service_timeout_secs)

        client_proc = await run_subprocess(
            [
                str(PYTHON),
                str(CLIENT),
                "--offer-url",
                f"http://127.0.0.1:{args.bot_port}/api/offer",
                "--audio",
                str(args.audio),
                "--run-secs",
                str(args.client_run_secs),
                "--silence-secs",
                str(args.client_silence_secs),
            ]
        )
        output, _ = await client_proc.communicate()
        if client_proc.returncode != 0:
            raise RuntimeError(f"aiortc client failed:\n{output.decode(errors='replace')}")

        log_patterns = ["TranscriptionFrame", "LLMTextFrame", "Bot started speaking"]
        stage_name = "Stage 1"
        if args.require_smart_turn:
            stage_name = "Stage 2"
            log_patterns.extend(
                [
                    "AudioOnlySmartTurnStopStrategy",
                    "UserStoppedSpeakingFrame",
                    "Added user audio turn to LLM context",
                    "sending context with",
                    "audio parts",
                ]
            )
        if args.require_local_tts:
            stage_name = "Stage 3"
            if args.tts_provider == "pocket":
                log_patterns.extend(
                    [
                        "Using local Kyutai Pocket TTS",
                        "PocketTTSService",
                        "local Pocket TTS stream complete",
                    ]
                )
            else:
                log_patterns.extend(
                    [
                        "Using local NVIDIA Magpie WebSocket TTS",
                        "NemotronMagpieWebSocketTTSService",
                        "TTSAudioRawFrame",
                        "local Magpie TTS stream complete",
                    ]
                )

        await wait_log_patterns(log_path, log_patterns, args.log_timeout_secs)

        print(f"{stage_name} smoke test passed. Log: {log_path}")
    finally:
        if client_proc:
            await terminate(client_proc)
        if bot_proc:
            await terminate(bot_proc)
        if asr_runner:
            await asr_runner.cleanup()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-mode", choices=["stub", "real"], default="stub")
    parser.add_argument("--asr-url", default="ws://127.0.0.1:8080")
    parser.add_argument("--asr-health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--asr-host", default="127.0.0.1")
    parser.add_argument("--asr-port", type=int, default=8088)
    parser.add_argument(
        "--stub-transcript",
        default="Tell me a story about a unicorn.",
    )
    parser.add_argument("--bot-host", default="0.0.0.0")
    parser.add_argument("--bot-port", type=int, default=7861)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--log", type=Path, default=ROOT / "nemotron-omni-step1-smoke.log")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--client-run-secs", type=float, default=30.0)
    parser.add_argument("--client-silence-secs", type=float, default=5.0)
    parser.add_argument("--service-timeout-secs", type=float, default=60.0)
    parser.add_argument("--log-timeout-secs", type=float, default=20.0)
    parser.add_argument("--vllm-health-url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--skip-vllm-health", action="store_true")
    parser.add_argument("--tts-provider", choices=["pocket", "magpie"], default="pocket")
    parser.add_argument(
        "--require-smart-turn",
        action="store_true",
        help="Require log evidence that Smart Turn finalized the user turn.",
    )
    parser.add_argument(
        "--require-local-tts",
        action="store_true",
        help="Require log evidence that the local NVIDIA Magpie TTS service produced speech.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except Exception as e:
        stage_name = "Stage 3" if args.require_local_tts else "Stage 2" if args.require_smart_turn else "Stage 1"
        print(f"{stage_name} smoke test failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
