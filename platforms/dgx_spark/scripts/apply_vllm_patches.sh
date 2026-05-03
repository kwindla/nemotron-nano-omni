#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

COMMON_PATCH="${NEMOTRON_REPO_ROOT}/platforms/common/patches/vllm-nemotron-omni-conversation-cache.patch"
SPARK_PATCH="${NEMOTRON_REPO_ROOT}/platforms/dgx_spark/patches/vllm-dgx-spark-weight-streaming.patch"

if [[ ! -d "${NEMOTRON_VLLM_SOURCE_DIR}/.git" ]]; then
  echo "Missing vLLM checkout at ${NEMOTRON_VLLM_SOURCE_DIR}" >&2
  exit 1
fi

for patch_file in "${COMMON_PATCH}" "${SPARK_PATCH}"; do
  if git -C "${NEMOTRON_VLLM_SOURCE_DIR}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "Patch already applied: ${patch_file}"
    continue
  fi
  git -C "${NEMOTRON_VLLM_SOURCE_DIR}" apply "${patch_file}"
  echo "Applied ${patch_file}"
done
