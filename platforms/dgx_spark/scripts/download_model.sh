#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/env.sh"

MODEL_REPO="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
MODEL_REVISION="${MODEL_REVISION:-12bcd2c16adf167fa2f3656f571c4e4313671be9}"
CRADIO_REPO="nvidia/C-RADIOv2-H"
CRADIO_REVISION="${CRADIO_REVISION:-0d8f4c18c877166eda07ddae1386bcad256b7a6a}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${NEMOTRON_MODEL_PATH}}"
RATE_LIMIT="${RATE_LIMIT:-100M}"

model_files=(
  "__init__.py"
  "audio_model.py"
  "chat_template.jinja"
  "config.json"
  "configuration.py"
  "configuration_nemotron_h.py"
  "configuration_radio.py"
  "evs.py"
  "generation_config.json"
  "image_processing.py"
  "model.safetensors.index.json"
  "modeling.py"
  "modeling_nemotron_h.py"
  "preprocessor_config.json"
  "processing.py"
  "processing_utils.py"
  "special_tokens_map.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "video_io.py"
  "video_processing.py"
  "media/2414-165385-0000.wav"
  "model-00001-of-00017.safetensors"
  "model-00002-of-00017.safetensors"
  "model-00003-of-00017.safetensors"
  "model-00004-of-00017.safetensors"
  "model-00005-of-00017.safetensors"
  "model-00006-of-00017.safetensors"
  "model-00007-of-00017.safetensors"
  "model-00008-of-00017.safetensors"
  "model-00009-of-00017.safetensors"
  "model-00010-of-00017.safetensors"
  "model-00011-of-00017.safetensors"
  "model-00012-of-00017.safetensors"
  "model-00013-of-00017.safetensors"
  "model-00014-of-00017.safetensors"
  "model-00015-of-00017.safetensors"
  "model-00016-of-00017.safetensors"
  "model-00017-of-00017.safetensors"
)

c_radio_files=(
  "adaptor_base.py"
  "adaptor_generic.py"
  "adaptor_mlp.py"
  "adaptor_registry.py"
  "cls_token.py"
  "common.py"
  "dinov2_arch.py"
  "dual_hybrid_vit.py"
  "enable_cpe_support.py"
  "enable_spectral_reparam.py"
  "eradio_model.py"
  "extra_models.py"
  "extra_timm_models.py"
  "feature_normalizer.py"
  "forward_intermediates.py"
  "hf_model.py"
  "input_conditioner.py"
  "open_clip_adaptor.py"
  "radio_model.py"
  "vit_patch_generator.py"
  "vitdet.py"
)

download_file() {
  local repo="$1"
  local revision="$2"
  local file="$3"
  local destination="${WEIGHTS_DIR}/${file}"

  mkdir -p "$(dirname "${destination}")"
  echo
  echo "==> ${repo}@${revision}: ${file}"
  curl \
    --fail \
    --location \
    --retry 20 \
    --retry-delay 5 \
    --retry-all-errors \
    --continue-at - \
    --limit-rate "${RATE_LIMIT}" \
    --output "${destination}" \
    "https://huggingface.co/${repo}/resolve/${revision}/${file}?download=true"
}

mkdir -p "${WEIGHTS_DIR}"
echo "Downloading ${MODEL_REPO} into ${WEIGHTS_DIR}"
echo "Bandwidth limit: ${RATE_LIMIT}; shards are downloaded sequentially."
echo "Pinned model revision: ${MODEL_REVISION}"
echo "Pinned C-RADIO revision: ${CRADIO_REVISION}"

for file in "${model_files[@]}"; do
  download_file "${MODEL_REPO}" "${MODEL_REVISION}" "${file}"
done

for file in "${c_radio_files[@]}"; do
  # Vendored into the model directory so trust-remote-code can stay local/offline.
  download_file "${CRADIO_REPO}" "${CRADIO_REVISION}" "${file}"
done

echo
echo "Download complete."
du -sh "${WEIGHTS_DIR}"
test -f "${WEIGHTS_DIR}/config.json"
test -f "${WEIGHTS_DIR}/model.safetensors.index.json"
