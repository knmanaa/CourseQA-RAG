#!/usr/bin/env bash
set -euo pipefail

derive_base_name() {
  local model_id="$1"
  local short_name
  short_name="${model_id##*/}"
  short_name="$(printf '%s' "${short_name}" | tr '[:upper:]' '[:lower:]')"
  short_name="$(printf '%s' "${short_name}" | sed -E 's/[^a-z0-9_-]+/_/g; s/^_+//; s/_+$//')"
  printf '%s\n' "${short_name}"
}

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODELS_DIR="${MODELS_DIR:-./models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
BASE_NAME="${BASE_NAME:-$(derive_base_name "${MODEL_ID}")}"
F16_GGUF="${F16_GGUF:-${MODELS_DIR}/${BASE_NAME}-f16.gguf}"
QUANTS="${QUANTS:-q8_0 q6_K q5_K_M q4_K_M q4_0}"

if [[ -x "${LLAMA_CPP_DIR}/llama-quantize" ]]; then
  LLAMA_QUANTIZE="${LLAMA_CPP_DIR}/llama-quantize"
elif [[ -x "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ]]; then
  LLAMA_QUANTIZE="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
else
  echo "Error: llama-quantize not found in ${LLAMA_CPP_DIR}. Build llama.cpp first." >&2
  exit 1
fi

if [[ ! -f "${F16_GGUF}" ]]; then
  echo "Error: missing base model ${F16_GGUF}" >&2
  echo "Run scripts/quantization/convert_to_gguf.sh first." >&2
  exit 1
fi

mkdir -p "${MODELS_DIR}"

for QUANT in ${QUANTS}; do
  OUTFILE="${MODELS_DIR}/${BASE_NAME}-${QUANT}.gguf"
  echo "Quantizing -> ${OUTFILE}"
  "${LLAMA_QUANTIZE}" "${F16_GGUF}" "${OUTFILE}" "${QUANT}"
done

echo "Done quantizing: ${BASE_NAME}"
