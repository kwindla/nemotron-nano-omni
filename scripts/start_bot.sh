#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${NEMOTRON_LOG_DIR:-${REPO_ROOT}/logs}"
PID_FILE="${NEMOTRON_BOT_PID:-${LOG_DIR}/bot.pid}"
STDOUT_LOG="${NEMOTRON_BOT_STDOUT_LOG:-${LOG_DIR}/bot.stdout.log}"
BOT_HOST="${NEMOTRON_BOT_HOST:-0.0.0.0}"
BOT_PORT="${NEMOTRON_BOT_PORT:-7860}"
BOT_TRANSPORT="${NEMOTRON_BOT_TRANSPORT:-webrtc}"
PYTHON_BIN="${NEMOTRON_BOT_PYTHON:-${REPO_ROOT}/.venv-pipecat/bin/python}"
BOT_ENTRY="${NEMOTRON_BOT_ENTRY:-${REPO_ROOT}/src/nemotron_voice/bot.py}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NEMOTRON_OMNI_LOG="${NEMOTRON_OMNI_LOG:-${LOG_DIR}/bot.log}"
export NEMOTRON_OMNI_BASE_URL="${NEMOTRON_OMNI_BASE_URL:-http://127.0.0.1:8000/v1}"
export NEMOTRON_SPEECH_STT_URL="${NEMOTRON_SPEECH_STT_URL:-ws://127.0.0.1:8080}"
export NEMOTRON_TTS_PROVIDER="${NEMOTRON_TTS_PROVIDER:-pocket}"
export POCKET_TTS_URL="${POCKET_TTS_URL:-http://127.0.0.1:8001}"

run_bot() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Pipecat Python at ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${BOT_ENTRY}" ]]; then
    echo "Missing bot entrypoint at ${BOT_ENTRY}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"
  exec "${PYTHON_BIN}" "${BOT_ENTRY}" \
    -t "${BOT_TRANSPORT}" \
    --host "${BOT_HOST}" \
    --port "${BOT_PORT}"
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

listening_pid_for_port() {
  ss -ltnp "( sport = :${BOT_PORT} )" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | head -n 1
}

pid_looks_like_bot() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  ps -p "${pid}" -o args= 2>/dev/null | grep -Fq "${BOT_ENTRY}"
}

port_owned_by_pid() {
  local pid
  pid="$(listening_pid_for_port)"
  [[ -n "${pid}" && -f "${PID_FILE}" && "${pid}" == "$(cat "${PID_FILE}")" ]]
}

http_ready() {
  curl --silent --show-error --fail \
    --max-time 1 \
    "http://127.0.0.1:${BOT_PORT}/openapi.json" >/dev/null 2>&1
}

command="${1:-start}"

case "${command}" in
  start)
    mkdir -p "${LOG_DIR}"
    if is_running; then
      echo "Bot already running with pid $(cat "${PID_FILE}")"
      exit 0
    fi
    existing_listener_pid="$(listening_pid_for_port || true)"
    if [[ -n "${existing_listener_pid}" ]]; then
      if pid_looks_like_bot "${existing_listener_pid}"; then
        echo "Bot port ${BOT_PORT} is already in use by pid ${existing_listener_pid}, but ${PID_FILE} is missing or stale." >&2
      else
        echo "Port ${BOT_PORT} is already in use by non-bot pid ${existing_listener_pid}." >&2
      fi
      echo "Stop the existing process or choose a different NEMOTRON_BOT_PORT before starting." >&2
      exit 1
    fi
    rm -f "${PID_FILE}"
    command_string="$(printf \
      'echo $$ > %q; cd %q; exec %q %q -t %q --host %q --port %q >>%q 2>&1 < /dev/null' \
      "${PID_FILE}" \
      "${REPO_ROOT}" \
      "${PYTHON_BIN}" \
      "${BOT_ENTRY}" \
      "${BOT_TRANSPORT}" \
      "${BOT_HOST}" \
      "${BOT_PORT}" \
      "${STDOUT_LOG}"
    )"
    setsid -f bash -lc "${command_string}"
    for _ in $(seq 1 40); do
      if is_running && port_owned_by_pid && http_ready; then
        break
      fi
      sleep 0.5
    done
    if ! is_running || ! port_owned_by_pid || ! http_ready; then
      echo "Bot failed to start; inspect ${STDOUT_LOG}" >&2
      if [[ -f "${PID_FILE}" ]]; then
        echo "PID file points to $(cat "${PID_FILE}")" >&2
      fi
      listener_pid="$(listening_pid_for_port || true)"
      if [[ -n "${listener_pid}" ]]; then
        echo "Port ${BOT_PORT} is currently owned by pid ${listener_pid}" >&2
      fi
      tail -n 40 "${STDOUT_LOG}" >&2 || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    echo "Started bot with pid $(cat "${PID_FILE}")"
    echo "Offer URL: http://127.0.0.1:${BOT_PORT}/api/offer"
    echo "Bot stdout log: ${STDOUT_LOG}"
    echo "Bot debug log: ${NEMOTRON_OMNI_LOG}"
    ;;
  foreground)
    mkdir -p "${LOG_DIR}"
    run_bot
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}"
      echo "Stopped bot"
    else
      rm -f "${PID_FILE}"
      echo "Bot is not running"
    fi
    ;;
  status)
    if is_running; then
      echo "Bot is running with pid $(cat "${PID_FILE}")"
      echo "Offer URL: http://127.0.0.1:${BOT_PORT}/api/offer"
      echo "Bot stdout log: ${STDOUT_LOG}"
      echo "Bot debug log: ${NEMOTRON_OMNI_LOG}"
    elif listener_pid="$(listening_pid_for_port || true)" && [[ -n "${listener_pid}" ]]; then
      if pid_looks_like_bot "${listener_pid}"; then
        echo "Bot appears to be running on port ${BOT_PORT} with pid ${listener_pid}, but ${PID_FILE} is missing or stale."
        echo "Offer URL: http://127.0.0.1:${BOT_PORT}/api/offer"
      else
        echo "Port ${BOT_PORT} is in use by non-bot pid ${listener_pid}."
      fi
      exit 1
    else
      echo "Bot is not running"
      exit 1
    fi
    ;;
  *)
    echo "Usage: ${SCRIPT_PATH} {start|foreground|stop|status}" >&2
    exit 1
    ;;
esac
