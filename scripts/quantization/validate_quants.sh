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
      PROMPT) [[ -z "${PROMPT}" ]] && PROMPT="${value}" ;;
      N_PREDICT) [[ -z "${N_PREDICT}" ]] && N_PREDICT="${value}" ;;
      TEMP) [[ -z "${TEMP}" ]] && TEMP="${value}" ;;
    esac
  done < "${file_path}"
}

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

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/model_config.txt}"
MODEL_ID="${MODEL_ID:-}"
MODELS_DIR="${MODELS_DIR:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
BASE_NAME="${BASE_NAME:-}"
PROMPT="${PROMPT:-}"
N_PREDICT="${N_PREDICT:-}"
TEMP="${TEMP:-}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--config-file" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    CONFIG_FILE="${ARGS[i+1]}"
  fi
done

CONFIG_FILE="$(resolve_repo_path "${CONFIG_FILE}")"
load_config_file "${CONFIG_FILE}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODELS_DIR="${MODELS_DIR:-./models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
PROMPT="${PROMPT:-Explain gradient descent in 3 concise bullet points.}"
N_PREDICT="${N_PREDICT:-128}"
TEMP="${TEMP:-0.1}"

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
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --n-predict)
      N_PREDICT="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'HELP'
Usage: validate_quants.sh [options]

Options:
  --config-file <path>  Config txt file (default: scripts/quantization/model_config.txt)
  --model-id <id>       Hugging Face model id (default: Qwen/Qwen3.5-9B)
  --models-dir <path>   Models directory (default: ./models)
  --llama-dir <path>    llama.cpp directory (default: ./llama.cpp)
  --base-name <name>    Base name (default: derived from --model-id)
  --prompt <text>       Validation prompt
  --n-predict <int>     Number of generated tokens
  --temp <float>        Sampling temperature
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

cleanup_llama_processes() {
  local bins=()
  local pids=()
  local pid=""

  if [[ -x "${LLAMA_CPP_DIR}/llama-cli" ]]; then
    bins+=("${LLAMA_CPP_DIR}/llama-cli")
  fi
  if [[ -x "${LLAMA_CPP_DIR}/build/bin/llama-cli" ]]; then
    bins+=("${LLAMA_CPP_DIR}/build/bin/llama-cli")
  fi
  if [[ -x "${LLAMA_CPP_DIR}/llama-completion" ]]; then
    bins+=("${LLAMA_CPP_DIR}/llama-completion")
  fi
  if [[ -x "${LLAMA_CPP_DIR}/build/bin/llama-completion" ]]; then
    bins+=("${LLAMA_CPP_DIR}/build/bin/llama-completion")
  fi

  for bin in "${bins[@]}"; do
    while IFS= read -r pid; do
      [[ -n "${pid}" && "${pid}" != "$$" ]] && pids+=("${pid}")
    done < <(pgrep -u "$(id -u)" -f "^${bin}( |$)" || true)
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No lingering llama validation processes found."
    return
  fi

  echo "Releasing memory: stopping lingering llama processes: ${pids[*]}"
  kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  sleep 1

  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done

  sync
  echo "Memory cleanup complete."
}

MODELS_DIR="$(resolve_repo_path "${MODELS_DIR}")"
LLAMA_CPP_DIR="$(resolve_repo_path "${LLAMA_CPP_DIR}")"

trap cleanup_llama_processes EXIT

if [[ -x "${LLAMA_CPP_DIR}/llama-cli" ]]; then
  LLAMA_CLI="${LLAMA_CPP_DIR}/llama-cli"
elif [[ -x "${LLAMA_CPP_DIR}/build/bin/llama-cli" ]]; then
  LLAMA_CLI="${LLAMA_CPP_DIR}/build/bin/llama-cli"
else
  echo "Error: llama-cli not found in ${LLAMA_CPP_DIR}. Build llama.cpp first." >&2
  exit 1
fi

shopt -s nullglob
MODELS=("${MODELS_DIR}/${BASE_NAME}-"*.gguf)
shopt -u nullglob

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Error: no GGUF files found matching ${MODELS_DIR}/${BASE_NAME}-*.gguf" >&2
  echo "Run scripts/quantization/quantize_all.sh first." >&2
  exit 1
fi

TOTAL_MODELS="${#MODELS[@]}"
INDEX=0

for GGUF in "${MODELS[@]}"; do
  INDEX=$((INDEX + 1))
  echo
  echo "=============================="
  echo "Model [${INDEX}/${TOTAL_MODELS}]: ${GGUF}"
  echo "=============================="
  "${LLAMA_CLI}" -m "${GGUF}" -p "${PROMPT}" -n "${N_PREDICT}" --temp "${TEMP}" -st </dev/null
done

echo

echo "Validation run complete."
