#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

BASE_URL="${NEMOTRON_VLLM_BASE_URL}"
LOG_DIR="${NEMOTRON_REPO_ROOT}/logs"
TEXT_OUT="${TEXT_OUT:-${LOG_DIR}/dgx-spark-text-smoke.json}"
AUDIO_OUT="${AUDIO_OUT:-${LOG_DIR}/dgx-spark-audio-smoke.json}"
TEXT_MAX_TOKENS="${TEXT_MAX_TOKENS:-256}"
AUDIO_MAX_TOKENS="${AUDIO_MAX_TOKENS:-512}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${NEMOTRON_VLLM_PYTHON}" ]]; then
  echo "Missing vLLM Python at ${NEMOTRON_VLLM_PYTHON}" >&2
  echo "Run platforms/dgx_spark/scripts/build_vllm.sh first or override NEMOTRON_VLLM_PYTHON." >&2
  exit 1
fi

for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

curl -fsS "${BASE_URL}/models" >/dev/null
curl -fsS "${BASE_URL}/models" | tee "${LOG_DIR}/dgx-spark-models.json" | jq .

curl -fsS "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "${NEMOTRON_VLLM_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "Reply with one short sentence confirming the text path is working."
    }
  ],
  "temperature": 0.0,
  "top_k": 1,
  "max_tokens": ${TEXT_MAX_TOKENS}
}
JSON
)" | tee "${TEXT_OUT}" | jq -r '.choices[0].message.content // .choices[0].message.reasoning'

AUDIO_URI="$("${NEMOTRON_VLLM_PYTHON}" - <<PY
import os
from pathlib import Path

print(Path(os.path.abspath("${NEMOTRON_AUDIO_FIXTURE}")).as_uri())
PY
)"

"${NEMOTRON_VLLM_PYTHON}" - <<PY >"${LOG_DIR}/dgx-spark-audio-payload.json"
import json

payload = {
    "model": "${NEMOTRON_VLLM_MODEL}",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "${AUDIO_URI}"}},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ],
    "temperature": 0.0,
    "top_k": 1,
    "max_tokens": ${AUDIO_MAX_TOKENS},
}
print(json.dumps(payload))
PY

curl -fsS "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  --data @"${LOG_DIR}/dgx-spark-audio-payload.json" \
  | tee "${AUDIO_OUT}" \
  | jq -r '.choices[0].message.content // .choices[0].message.reasoning'
