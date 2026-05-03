#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_SCRIPT="${SCRIPT_DIR}/start_client_dev.sh"
BOT_SCRIPT="${SCRIPT_DIR}/start_bot.sh"

command="${1:-start}"

case "${command}" in
  start)
    "${BOT_SCRIPT}" start
    "${CLIENT_SCRIPT}" start
    ;;
  stop)
    "${CLIENT_SCRIPT}" stop || true
    "${BOT_SCRIPT}" stop || true
    ;;
  restart)
    "${CLIENT_SCRIPT}" stop || true
    "${BOT_SCRIPT}" stop || true
    "${BOT_SCRIPT}" start
    "${CLIENT_SCRIPT}" start
    ;;
  status)
    "${BOT_SCRIPT}" status
    "${CLIENT_SCRIPT}" status
    ;;
  install-client)
    "${CLIENT_SCRIPT}" install
    ;;
  *)
    echo "Usage: ${0} {start|stop|restart|status|install-client}" >&2
    exit 1
    ;;
esac
