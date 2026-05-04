#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VLLM_REF="${VLLM_REF:-88d34c6409e9fb3c7b8ca0c04756f061d2099eb1}"
RECREATE_ENV="${NEMOTRON_VLLM_RECREATE_ENV:-0}"
USE_PRECOMPILED="${NEMOTRON_VLLM_USE_PRECOMPILED:-1}"
PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-${VLLM_REF}}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export MAX_JOBS="${MAX_JOBS:-6}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-500}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  detected_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ -n "${detected_cap}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${detected_cap}+PTX"
  fi
fi

if [[ ! -d "${NEMOTRON_VLLM_SOURCE_DIR}" ]]; then
  echo "vLLM source tree not found at ${NEMOTRON_VLLM_SOURCE_DIR}" >&2
  exit 1
fi

validate_env() {
  local python_bin="${NEMOTRON_VLLM_VENV}/bin/python3"
  if [[ ! -x "${python_bin}" ]]; then
    echo "vLLM env does not contain ${python_bin}" >&2
    return 1
  fi

  PYTHONPATH="${NEMOTRON_VLLM_SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - <<'PY'
import importlib.metadata as md
from pathlib import Path

import timm
import torch
import transformers
import vllm
import vllm._C
import vllm.cumem_allocator

print("python", __import__("sys").version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", md.version("vllm"))
print("timm", md.version("timm"))
print("vllm_source", Path(vllm.__file__).resolve())
PY
}

echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<default>}"
if [[ -d "${NEMOTRON_VLLM_SOURCE_DIR}/.git" ]]; then
  current_ref="$(git -C "${NEMOTRON_VLLM_SOURCE_DIR}" rev-parse HEAD)"
  echo "Expected vLLM ref: ${VLLM_REF}"
  echo "Current vLLM ref:  ${current_ref}"
  if [[ "${current_ref}" != "${VLLM_REF}" ]]; then
    echo "WARNING: vLLM checkout does not match VLLM_REF." >&2
  fi
fi

if [[ "${RECREATE_ENV}" == "1" && -d "${NEMOTRON_VLLM_VENV}" ]]; then
  echo "Removing existing vLLM env at ${NEMOTRON_VLLM_VENV}"
  rm -rf "${NEMOTRON_VLLM_VENV}"
fi

uv venv "${NEMOTRON_VLLM_VENV}" --python "${PYTHON_BIN}" --allow-existing

# shellcheck disable=SC1090
source "${NEMOTRON_VLLM_VENV}/bin/activate"

cd "${NEMOTRON_VLLM_SOURCE_DIR}"
if [[ "${USE_PRECOMPILED}" == "1" ]]; then
  export VLLM_USE_PRECOMPILED=1
  export VLLM_PRECOMPILED_WHEEL_COMMIT="${PRECOMPILED_WHEEL_COMMIT}"
  echo "Installing vLLM with precompiled extension wheel for commit ${VLLM_PRECOMPILED_WHEEL_COMMIT}"
else
  unset VLLM_USE_PRECOMPILED
  unset VLLM_PRECOMPILED_WHEEL_COMMIT
  echo "Installing vLLM with a local source build"
fi

uv pip install --editable ".[audio]" --torch-backend "${UV_TORCH_BACKEND}"
uv pip install timm

validate_env
