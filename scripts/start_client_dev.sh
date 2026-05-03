#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLIENT_DIR="${NEMOTRON_CLIENT_DIR:-${REPO_ROOT}/client}"
LOG_DIR="${NEMOTRON_LOG_DIR:-${REPO_ROOT}/logs}"
PID_FILE="${NEMOTRON_CLIENT_PID:-${LOG_DIR}/client-dev.pid}"
LOG_FILE="${NEMOTRON_CLIENT_LOG:-${LOG_DIR}/client-dev.log}"
HOST="${NEMOTRON_CLIENT_HOST:-127.0.0.1}"
PORT="${NEMOTRON_CLIENT_PORT:-5173}"
BOT_PORT="${NEMOTRON_BOT_PORT:-7860}"
INSTALL_MODE="${NEMOTRON_CLIENT_INSTALL_MODE:-auto}"
export PIPECAT_BOT_URL="${PIPECAT_BOT_URL:-http://127.0.0.1:${BOT_PORT}}"

pnpm_cmd=()

resolve_pnpm() {
  if command -v pnpm >/dev/null 2>&1; then
    pnpm_cmd=(pnpm)
    return
  fi
  if command -v corepack >/dev/null 2>&1; then
    pnpm_cmd=(corepack pnpm)
    return
  fi
  echo "Missing pnpm and corepack; install one of them first." >&2
  exit 1
}

run_install() {
  resolve_pnpm
  cd "${CLIENT_DIR}"
  "${pnpm_cmd[@]}" install
}

ensure_dependencies() {
  case "${INSTALL_MODE}" in
    always)
      run_install
      ;;
    auto)
      if [[ ! -d "${CLIENT_DIR}/node_modules" ]]; then
        run_install
      fi
      ;;
    never)
      ;;
    *)
      echo "Unsupported NEMOTRON_CLIENT_INSTALL_MODE=${INSTALL_MODE}" >&2
      exit 1
      ;;
  esac
}

run_server() {
  if [[ ! -d "${CLIENT_DIR}" ]]; then
    echo "Missing client directory at ${CLIENT_DIR}" >&2
    exit 1
  fi

  ensure_dependencies
  resolve_pnpm
  cd "${CLIENT_DIR}"
  exec "${pnpm_cmd[@]}" run dev -- --host "${HOST}" --port "${PORT}"
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

command="${1:-start}"

case "${command}" in
  install)
    mkdir -p "${LOG_DIR}"
    run_install
    ;;
  start)
    mkdir -p "${LOG_DIR}"
    if is_running; then
      echo "Client dev server already running with pid $(cat "${PID_FILE}")"
      exit 0
    fi
    ensure_dependencies
    resolve_pnpm
    rm -f "${PID_FILE}"
    printf -v pnpm_exec '%q ' "${pnpm_cmd[@]}"
    command_string="$(printf \
      'echo $$ > %q; cd %q; exec %srun dev -- --host %q --port %q >>%q 2>&1 < /dev/null' \
      "${PID_FILE}" \
      "${CLIENT_DIR}" \
      "${pnpm_exec}" \
      "${HOST}" \
      "${PORT}" \
      "${LOG_FILE}"
    )"
    setsid -f bash -lc "${command_string}"
    for _ in $(seq 1 20); do
      if is_running; then
        break
      fi
      sleep 0.5
    done
    if ! is_running; then
      echo "Client dev server failed to start; inspect ${LOG_FILE}" >&2
      tail -n 40 "${LOG_FILE}" >&2 || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    echo "Started client dev server with pid $(cat "${PID_FILE}")"
    echo "URL: http://${HOST}:${PORT}"
    echo "Log: ${LOG_FILE}"
    ;;
  foreground)
    mkdir -p "${LOG_DIR}"
    run_server
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}"
      echo "Stopped client dev server"
    else
      rm -f "${PID_FILE}"
      echo "Client dev server is not running"
    fi
    ;;
  status)
    if is_running; then
      echo "Client dev server is running with pid $(cat "${PID_FILE}")"
      echo "URL: http://${HOST}:${PORT}"
      echo "Proxy target: ${PIPECAT_BOT_URL}"
      echo "Log: ${LOG_FILE}"
    else
      echo "Client dev server is not running"
      exit 1
    fi
    ;;
  *)
    echo "Usage: ${SCRIPT_PATH} {install|start|foreground|stop|status}" >&2
    exit 1
    ;;
esac
