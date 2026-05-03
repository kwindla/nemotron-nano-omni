#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

COMMON_PATCH="${NEMOTRON_REPO_ROOT}/platforms/common/patches/vllm-nemotron-omni-conversation-cache.patch"

if [[ ! -d "${NEMOTRON_VLLM_SOURCE_DIR}/.git" ]]; then
  echo "Missing vLLM checkout at ${NEMOTRON_VLLM_SOURCE_DIR}" >&2
  exit 1
fi

if git -C "${NEMOTRON_VLLM_SOURCE_DIR}" apply --reverse --check "${COMMON_PATCH}" >/dev/null 2>&1; then
  echo "Common vLLM patch already applied."
  exit 0
fi

git -C "${NEMOTRON_VLLM_SOURCE_DIR}" apply "${COMMON_PATCH}"
echo "Applied ${COMMON_PATCH}"
