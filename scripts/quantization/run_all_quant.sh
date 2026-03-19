#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

resolve_repo_path() {
  local input_path="$1"
  if [[ "${input_path}" = /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${REPO_ROOT}/${input_path#./}"
  fi
}

derive_base_name() {
  local model_id="$1"
  local short_name
  short_name="${model_id##*/}"
  short_name="$(printf '%s' "${short_name}" | tr '[:upper:]' '[:lower:]')"
  short_name="$(printf '%s' "${short_name}" | sed -E 's/[^a-z0-9_-]+/_/g; s/^_+//; s/_+$//')"
  printf '%s\n' "${short_name}"
}

load_config_file() {
  local file_path="$1"
  local key=""
  local value=""

  [[ -f "${file_path}" ]] || return 0

  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%%[[:space:]]*}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue

    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"

    if [[ "${value}" =~ ^".*"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" =~ ^'.*'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    case "${key}" in
      MODEL_ID) [[ -z "${MODEL_ID}" ]] && MODEL_ID="${value}" ;;
      MODELS_DIR) [[ -z "${MODELS_DIR}" ]] && MODELS_DIR="${value}" ;;
      LLAMA_CPP_DIR) [[ -z "${LLAMA_CPP_DIR}" ]] && LLAMA_CPP_DIR="${value}" ;;
      BASE_NAME) [[ -z "${BASE_NAME}" ]] && BASE_NAME="${value}" ;;
      QUANTS) [[ -z "${QUANTS}" ]] && QUANTS="${value}" ;;
      ENV_NAME) [[ -z "${ENV_NAME}" ]] && ENV_NAME="${value}" ;;
      HF_TOKEN) [[ -z "${HF_TOKEN}" ]] && HF_TOKEN="${value}" ;;
    esac
  done < "${file_path}"
}

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/model_config.txt}"
MODEL_ID="${MODEL_ID:-}"
MODELS_DIR="${MODELS_DIR:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
BASE_NAME="${BASE_NAME:-}"
QUANTS="${QUANTS:-}"
FRESH="${FRESH:-0}"
ENV_NAME="${ENV_NAME:-}"
INSIDE_CONDA_RUN="${INSIDE_CONDA_RUN:-0}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--config-file" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    CONFIG_FILE="${ARGS[i+1]}"
  fi
done

CONFIG_FILE="$(resolve_repo_path "${CONFIG_FILE}")"
load_config_file "${CONFIG_FILE}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/llama.cpp}"
QUANTS="${QUANTS:-q8_0 q6_K q5_K_M q4_K_M q4_0}"
ENV_NAME="${ENV_NAME:-CourseQARAG}"

set -- "${ARGS[@]}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --llama-dir)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --base-name)
      BASE_NAME="$2"
      shift 2
      ;;
    --quants)
      QUANTS="$2"
      shift 2
      ;;
    --fresh)
      FRESH="1"
      shift
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --inside-conda-run)
      INSIDE_CONDA_RUN="1"
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: run_all_quant.sh [options]

Runs full quantization pipeline:
  1) convert_to_gguf.sh
  2) quantize_all.sh

Validation is intentionally separate via validate_quants.sh.

Defaults are for Qwen/Qwen3.5-9B.

Options:
  --config-file <path>  Config txt file (default: scripts/quantization/model_config.txt)
  --model-id <id>       Hugging Face model id (default: Qwen/Qwen3.5-9B)
  --models-dir <path>   Models directory (default: ./models)
  --llama-dir <path>    llama.cpp directory (default: ./llama.cpp)
  --base-name <name>    Base output name (default: derived from --model-id)
  --quants "..."        Space-separated quant list
  --fresh               Remove previous artifacts/logs for this base model before run
  --env <name>          Run inside specified conda env (default: CourseQARAG)
  --hf-token <token>    Hugging Face token (for gated/private models)
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BASE_NAME}" ]]; then
  BASE_NAME="$(derive_base_name "${MODEL_ID}")"
fi

CONFIG_FILE="$(resolve_repo_path "${CONFIG_FILE}")"
MODELS_DIR="$(resolve_repo_path "${MODELS_DIR}")"
LLAMA_CPP_DIR="$(resolve_repo_path "${LLAMA_CPP_DIR}")"

export MODEL_ID BASE_NAME QUANTS MODELS_DIR LLAMA_CPP_DIR
if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
fi

cd "${REPO_ROOT}"

if [[ -n "${ENV_NAME}" && "${INSIDE_CONDA_RUN}" != "1" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found in PATH (required for --env)." >&2
    exit 1
  fi

  exec env HF_TOKEN="${HF_TOKEN}" conda run --no-capture-output -n "${ENV_NAME}" "${SCRIPT_DIR}/run_all_quant.sh" \
    --inside-conda-run \
    --config-file "${CONFIG_FILE}" \
    --model-id "${MODEL_ID}" \
    --models-dir "${MODELS_DIR}" \
    --llama-dir "${LLAMA_CPP_DIR}" \
    --base-name "${BASE_NAME}" \
    --quants "${QUANTS}" \
    $( [[ "${FRESH}" == "1" ]] && printf '%s' '--fresh' )
fi

if [[ ! -x "${SCRIPT_DIR}/convert_to_gguf.sh" || ! -x "${SCRIPT_DIR}/quantize_all.sh" ]]; then
  echo "Error: one or more required scripts are missing/executable in ${SCRIPT_DIR}" >&2
  exit 1
fi

HF_LOCAL_DIR="${MODELS_DIR}/${BASE_NAME}-hf"
OUTFILE="${MODELS_DIR}/${BASE_NAME}-f16.gguf"

if [[ "${MODEL_ID}" == "google/gemma-3-27b-it" && -z "${HF_TOKEN}" ]]; then
  echo "Error: ${MODEL_ID} is gated. Set HF_TOKEN in config or pass --hf-token." >&2
  exit 1
fi

model_artifacts_exist="0"
if [[ -d "${HF_LOCAL_DIR}" || -f "${OUTFILE}" ]]; then
  model_artifacts_exist="1"
else
  for quant in ${QUANTS}; do
    if [[ -f "${MODELS_DIR}/${BASE_NAME}-${quant}.gguf" ]]; then
      model_artifacts_exist="1"
      break
    fi
  done
fi

if [[ "${FRESH}" == "1" ]]; then
  if [[ "${model_artifacts_exist}" == "1" ]]; then
    echo "== [0/2] Cleaning previous artifacts for ${BASE_NAME} =="
    rm -rf "${HF_LOCAL_DIR}" "${OUTFILE}"
    for quant in ${QUANTS}; do
      rm -f "${MODELS_DIR}/${BASE_NAME}-${quant}.gguf"
    done
  else
    echo "== [0/2] No prior artifacts found for ${BASE_NAME}; skipping cleanup =="
  fi
elif [[ "${model_artifacts_exist}" == "1" ]]; then
  echo "== [0/2] Existing artifacts detected for ${BASE_NAME}; preserving files (no cleanup) =="
fi

echo "== [1/2] Convert HF -> GGUF =="
MODEL_ID="${MODEL_ID}" \
MODELS_DIR="${MODELS_DIR}" \
LLAMA_CPP_DIR="${LLAMA_CPP_DIR}" \
HF_LOCAL_DIR="${HF_LOCAL_DIR}" \
OUTFILE="${OUTFILE}" \
OUTTYPE="f16" \
HF_TOKEN="${HF_TOKEN}" \
"${SCRIPT_DIR}/convert_to_gguf.sh"

echo "== [2/2] Quantize GGUF =="
MODELS_DIR="${MODELS_DIR}" \
LLAMA_CPP_DIR="${LLAMA_CPP_DIR}" \
BASE_NAME="${BASE_NAME}" \
F16_GGUF="${OUTFILE}" \
QUANTS="${QUANTS}" \
"${SCRIPT_DIR}/quantize_all.sh"

echo
echo "Quantization steps completed for ${MODEL_ID}."
echo "Run validation separately with:"
echo "  MODEL_ID=\"${MODEL_ID}\" MODELS_DIR=\"${MODELS_DIR}\" LLAMA_CPP_DIR=\"${LLAMA_CPP_DIR}\" BASE_NAME=\"${BASE_NAME}\" ${SCRIPT_DIR}/validate_quants.sh"
