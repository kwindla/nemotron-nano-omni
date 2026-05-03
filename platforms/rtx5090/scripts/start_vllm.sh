#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/../config/env.sh"

LOG_DIR="$(dirname "${NEMOTRON_VLLM_LOG}")"
PID_FILE="${NEMOTRON_VLLM_PID}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"

run_server() {
  if [[ ! -x "${NEMOTRON_VLLM_BIN}" ]]; then
    echo "Missing vLLM executable at ${NEMOTRON_VLLM_BIN}" >&2
    echo "Build or point NEMOTRON_VLLM_BIN at a valid vLLM install." >&2
    exit 1
  fi

  if [[ -d "${NEMOTRON_VLLM_VENV}" ]]; then
    # Keep the platform wrapper aligned with the installed environment.
    # shellcheck disable=SC1090
    source "${NEMOTRON_VLLM_VENV}/bin/activate"
  fi

  cd "${NEMOTRON_REPO_ROOT}"

  local -a extra_args=()
  if [[ "${NEMOTRON_VLLM_ENFORCE_EAGER}" == "1" ]]; then
    extra_args+=(--enforce-eager)
  fi
  if [[ "${NEMOTRON_VLLM_ENABLE_PREFIX_CACHING}" == "1" ]]; then
    extra_args+=(--enable-prefix-caching)
  else
    extra_args+=(--no-enable-prefix-caching)
  fi
  if [[ -n "${NEMOTRON_VLLM_KV_CACHE_MEMORY_BYTES}" ]]; then
    extra_args+=(--kv-cache-memory-bytes "${NEMOTRON_VLLM_KV_CACHE_MEMORY_BYTES}")
  fi
  if [[ "${NEMOTRON_VLLM_KV_CACHE_DTYPE}" != "auto" ]]; then
    extra_args+=(--kv-cache-dtype "${NEMOTRON_VLLM_KV_CACHE_DTYPE}")
  fi
  if [[ "${NEMOTRON_VLLM_SKIP_MM_PROFILING}" == "1" ]]; then
    extra_args+=(--skip-mm-profiling)
  fi
  if [[ "${NEMOTRON_VLLM_ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    extra_args+=(--enable-auto-tool-choice)
  fi
  if [[ -n "${NEMOTRON_VLLM_REASONING_PARSER}" ]]; then
    extra_args+=(--reasoning-parser "${NEMOTRON_VLLM_REASONING_PARSER}")
  fi
  if [[ -n "${NEMOTRON_VLLM_TOOL_CALL_PARSER}" ]]; then
    extra_args+=(--tool-call-parser "${NEMOTRON_VLLM_TOOL_CALL_PARSER}")
  fi
  if [[ -n "${NEMOTRON_VLLM_MOE_BACKEND}" && "${NEMOTRON_VLLM_MOE_BACKEND}" != "auto" ]]; then
    extra_args+=(--moe-backend "${NEMOTRON_VLLM_MOE_BACKEND}")
  fi
  if [[ -n "${NEMOTRON_VLLM_ATTENTION_BACKEND}" && "${NEMOTRON_VLLM_ATTENTION_BACKEND}" != "auto" ]]; then
    extra_args+=(--attention-backend "${NEMOTRON_VLLM_ATTENTION_BACKEND}")
  fi
  if [[ -n "${NEMOTRON_VLLM_MAMBA_CACHE_MODE}" ]]; then
    extra_args+=(--mamba-cache-mode "${NEMOTRON_VLLM_MAMBA_CACHE_MODE}")
  fi
  if [[ -n "${NEMOTRON_VLLM_MAMBA_BACKEND}" ]]; then
    extra_args+=(--mamba-backend "${NEMOTRON_VLLM_MAMBA_BACKEND}")
  fi
  if [[ -n "${NEMOTRON_VLLM_MM_ENCODER_ATTN_BACKEND}" ]]; then
    extra_args+=(--mm-encoder-attn-backend "${NEMOTRON_VLLM_MM_ENCODER_ATTN_BACKEND}")
  fi

  export VLLM_CONVERSATION_CACHE_MAX_ENTRIES
  export VLLM_CONVERSATION_CACHE_MAX_TOKENS
  export VLLM_CONVERSATION_CACHE_MIN_FREE_BLOCKS
  export VLLM_CONVERSATION_CACHE_TARGET_FREE_BLOCKS

  exec "${NEMOTRON_VLLM_BIN}" serve "${NEMOTRON_MODEL_PATH}" \
    --served-model-name "${NEMOTRON_VLLM_MODEL}" \
    --host "${NEMOTRON_VLLM_HOST}" \
    --port "${NEMOTRON_VLLM_PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization "${NEMOTRON_VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${NEMOTRON_VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${NEMOTRON_VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${NEMOTRON_VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --limit-mm-per-prompt "${NEMOTRON_VLLM_LIMIT_MM_PER_PROMPT}" \
    --allowed-local-media-path "${NEMOTRON_VLLM_ALLOWED_LOCAL_MEDIA_PATH}" \
    "${extra_args[@]}"
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

command="${1:-start}"

case "${command}" in
  start)
    mkdir -p "${LOG_DIR}"
    if is_running; then
      echo "vLLM already running with pid $(cat "${PID_FILE}")"
      exit 0
    fi
    nohup "${SCRIPT_PATH}" foreground >"${NEMOTRON_VLLM_LOG}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "Started vLLM with pid $(cat "${PID_FILE}")"
    echo "Log: ${NEMOTRON_VLLM_LOG}"
    ;;
  foreground)
    mkdir -p "${LOG_DIR}"
    run_server
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}"
      echo "Stopped vLLM"
    else
      rm -f "${PID_FILE}"
      echo "vLLM is not running"
    fi
    ;;
  status)
    if is_running; then
      echo "vLLM is running with pid $(cat "${PID_FILE}")"
    else
      echo "vLLM is not running"
      exit 1
    fi
    ;;
  *)
    echo "Usage: ${SCRIPT_PATH} {start|foreground|stop|status}" >&2
    exit 1
    ;;
esac
