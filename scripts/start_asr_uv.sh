#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${NEMOTRON_SPEECH_PYTHON:=.venv-asr/bin/python}"
: "${NEMOTRON_SPEECH_HOST:=0.0.0.0}"
: "${NEMOTRON_SPEECH_PORT:=8080}"
: "${NEMOTRON_SPEECH_MODEL:=models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
: "${NEMOTRON_SPEECH_RIGHT_CONTEXT:=1}"
: "${NEMOTRON_SPEECH_DEVICE:=cuda}"

if [[ ! -x "$NEMOTRON_SPEECH_PYTHON" ]]; then
  echo "Missing ASR Python at $NEMOTRON_SPEECH_PYTHON" >&2
  echo "Create it with the README's .venv-asr setup commands." >&2
  exit 1
fi

exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$NEMOTRON_SPEECH_PYTHON" -m nemotron_speech.server \
    --host "$NEMOTRON_SPEECH_HOST" \
    --port "$NEMOTRON_SPEECH_PORT" \
    --model "$NEMOTRON_SPEECH_MODEL" \
    --right-context "$NEMOTRON_SPEECH_RIGHT_CONTEXT" \
    --device "$NEMOTRON_SPEECH_DEVICE"
