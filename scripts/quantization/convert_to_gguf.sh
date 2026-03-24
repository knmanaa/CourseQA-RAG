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
HF_LOCAL_DIR="${HF_LOCAL_DIR:-${MODELS_DIR}/${BASE_NAME}-hf}"
OUTFILE="${OUTFILE:-${MODELS_DIR}/${BASE_NAME}-f16.gguf}"
OUTTYPE="${OUTTYPE:-f16}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI_CMD="huggingface-cli"
elif command -v hf >/dev/null 2>&1; then
  HF_CLI_CMD="hf"
else
  echo "Error: neither huggingface-cli nor hf found in PATH. Install with: pip install huggingface_hub" >&2
  exit 1
fi

if ! python -c "import sentencepiece" >/dev/null 2>&1; then
  echo "Error: Python package 'sentencepiece' is required for HF->GGUF conversion." >&2
  echo "Install it in your active environment and retry:" >&2
  echo "  pip install sentencepiece" >&2
  exit 1
fi

mkdir -p "${MODELS_DIR}"

echo "[1/2] Downloading ${MODEL_ID} to ${HF_LOCAL_DIR}"
download_args=(download "${MODEL_ID}" --local-dir "${HF_LOCAL_DIR}")
if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
  download_args+=(--token "${HF_TOKEN}")
  echo "Using Hugging Face token from environment/config for gated access"
fi
"${HF_CLI_CMD}" "${download_args[@]}"

echo "[2/2] Converting HF -> GGUF (${OUTTYPE})"
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${HF_LOCAL_DIR}" \
  --outfile "${OUTFILE}" \
  --outtype "${OUTTYPE}"

echo "Done: ${OUTFILE}"
