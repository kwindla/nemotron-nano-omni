#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

cd "${NEMOTRON_REPO_ROOT}"

if [[ ! -x "${NEMOTRON_VLLM_PYTHON}" ]]; then
  echo "Missing vLLM Python at ${NEMOTRON_VLLM_PYTHON}" >&2
  echo "Build the platform vLLM env first or override NEMOTRON_VLLM_PYTHON." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- --reuse-server
fi

exec "${NEMOTRON_VLLM_PYTHON}" scripts/test_vllm_conversation_prefix_cache.py "$@"
