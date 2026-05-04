#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/../config/env.sh"

LOG_DIR="$(dirname "${NEMOTRON_VLLM_LOG}")"
PID_FILE="${NEMOTRON_VLLM_PID}"
READY_TIMEOUT_SECS="${NEMOTRON_VLLM_START_TIMEOUT_SECS}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"

validate_env() {
  if [[ ! -x "${NEMOTRON_VLLM_BIN}" ]]; then
    echo "Missing vLLM executable at ${NEMOTRON_VLLM_BIN}" >&2
    echo "Build or point NEMOTRON_VLLM_BIN at a valid vLLM install." >&2
    return 1
  fi
  if [[ ! -d "${NEMOTRON_MODEL_PATH}" ]]; then
    echo "Missing model directory at ${NEMOTRON_MODEL_PATH}" >&2
    return 1
  fi
}

run_server() {
  validate_env

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
  if [[ -n "${NEMOTRON_VLLM_MAMBA_CACHE_DTYPE}" && "${NEMOTRON_VLLM_MAMBA_CACHE_DTYPE}" != "auto" ]]; then
    extra_args+=(--mamba-cache-dtype "${NEMOTRON_VLLM_MAMBA_CACHE_DTYPE}")
  fi
  if [[ -n "${NEMOTRON_VLLM_MAMBA_SSM_CACHE_DTYPE}" && "${NEMOTRON_VLLM_MAMBA_SSM_CACHE_DTYPE}" != "auto" ]]; then
    extra_args+=(--mamba-ssm-cache-dtype "${NEMOTRON_VLLM_MAMBA_SSM_CACHE_DTYPE}")
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
  export VLLM_CONVERSATION_CACHE_FILTER_PREFIX_MM_INPUTS

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

remove_stale_pid_file() {
  if [[ -f "${PID_FILE}" ]] && ! is_running; then
    rm -f "${PID_FILE}"
  fi
}

port_listener() {
  ss -H -ltnp "( sport = :${NEMOTRON_VLLM_PORT} )" 2>/dev/null || true
}

is_ready() {
  local models_url="${NEMOTRON_VLLM_BASE_URL}/models"
  local response

  response="$(curl -fsS --max-time 5 "${models_url}" 2>/dev/null || true)"
  if [[ -z "${response}" ]]; then
    return 1
  fi

  python3 -c 'import json, sys; payload=json.loads(sys.stdin.read()); model=sys.argv[1]; sys.exit(0 if any(item.get("id") == model for item in payload.get("data", [])) else 1)' \
    "${NEMOTRON_VLLM_MODEL}" <<<"${response}"
}

wait_for_ready() {
  local pid="$1"
  local deadline=$((SECONDS + READY_TIMEOUT_SECS))

  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM exited before becoming ready." >&2
      return 1
    fi
    if is_ready; then
      return 0
    fi
    sleep 5
  done

  echo "Timed out after ${READY_TIMEOUT_SECS}s waiting for vLLM readiness." >&2
  return 1
}

wait_for_pid_file() {
  local deadline=$((SECONDS + 15))

  while (( SECONDS < deadline )); do
    if [[ -s "${PID_FILE}" ]]; then
      return 0
    fi
    sleep 0.2
  done

  echo "Timed out waiting for ${PID_FILE} to be written." >&2
  return 1
}

print_failure_context() {
  echo "Last 80 log lines from ${NEMOTRON_VLLM_LOG}:" >&2
  tail -n 80 "${NEMOTRON_VLLM_LOG}" >&2 || true
}

command="${1:-start}"

case "${command}" in
  start)
    mkdir -p "${LOG_DIR}"
    validate_env
    remove_stale_pid_file

    if is_running; then
      if is_ready; then
        echo "vLLM already running with pid $(cat "${PID_FILE}")"
        exit 0
      fi
      echo "vLLM process $(cat "${PID_FILE}") is running but not ready yet"
      exit 0
    fi

    if [[ -n "$(port_listener)" ]]; then
      echo "Port ${NEMOTRON_VLLM_PORT} is already in use:" >&2
      port_listener >&2
      exit 1
    fi

    if [[ -f "${NEMOTRON_VLLM_LOG}" ]]; then
      mv "${NEMOTRON_VLLM_LOG}" "${NEMOTRON_VLLM_LOG}.prev"
    fi

    rm -f "${PID_FILE}"
    setsid -f bash "${SCRIPT_PATH}" detached >"${NEMOTRON_VLLM_LOG}" 2>&1 < /dev/null

    if ! wait_for_pid_file; then
      print_failure_context
      exit 1
    fi

    if wait_for_ready "$(cat "${PID_FILE}")"; then
      echo "Started vLLM with pid $(cat "${PID_FILE}")"
      echo "Ready: ${NEMOTRON_VLLM_BASE_URL}/models"
      echo "Log: ${NEMOTRON_VLLM_LOG}"
    else
      kill "$(cat "${PID_FILE}")" 2>/dev/null || true
      wait "$(cat "${PID_FILE}")" 2>/dev/null || true
      rm -f "${PID_FILE}"
      print_failure_context
      exit 1
    fi
    ;;
  foreground)
    mkdir -p "${LOG_DIR}"
    run_server
    ;;
  detached)
    mkdir -p "${LOG_DIR}"
    echo $$ >"${PID_FILE}"
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
    remove_stale_pid_file
    if is_running; then
      if is_ready; then
        echo "vLLM is ready with pid $(cat "${PID_FILE}")"
      else
        echo "vLLM is running with pid $(cat "${PID_FILE}") but readiness has not passed yet"
        exit 2
      fi
    elif [[ -n "$(port_listener)" ]]; then
      echo "Port ${NEMOTRON_VLLM_PORT} is in use by an external process:" >&2
      port_listener >&2
      exit 1
    else
      echo "vLLM is not running"
      exit 1
    fi
    ;;
  check-env)
    validate_env
    ;;
  *)
    echo "Usage: ${SCRIPT_PATH} {start|foreground|detached|stop|status|check-env}" >&2
    exit 1
    ;;
esac
