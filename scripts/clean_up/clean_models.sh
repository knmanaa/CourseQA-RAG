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
      MODELS_DIR) [[ -z "${MODELS_DIR}" ]] && MODELS_DIR="${value}" ;;
      MODEL_ID) [[ -z "${MODEL_ID}" ]] && MODEL_ID="${value}" ;;
      BASE_NAME) [[ -z "${BASE_NAME}" ]] && BASE_NAME="${value}" ;;
      ASSUME_YES) [[ "${ASSUME_YES}" == "0" ]] && ASSUME_YES="${value}" ;;
      DRY_RUN) [[ "${DRY_RUN}" == "0" ]] && DRY_RUN="${value}" ;;
    esac
  done < "${file_path}"
}

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/clean_up_config.txt}"
MODELS_DIR="${MODELS_DIR:-}"
MODEL_ID="${MODEL_ID:-}"
BASE_NAME="${BASE_NAME:-}"
REMOVE_ALL="${REMOVE_ALL:-0}"
DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == "--config-file" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    CONFIG_FILE="${ARGS[i+1]}"
  fi
done

CONFIG_FILE="$(resolve_repo_path "${CONFIG_FILE}")"
load_config_file "${CONFIG_FILE}"

MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"

set -- "${ARGS[@]}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --base-name)
      BASE_NAME="$2"
      shift 2
      ;;
    --all)
      REMOVE_ALL="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --yes)
      ASSUME_YES="1"
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: clean_models.sh [options]

Cleans model artifacts under the models directory.

Default behavior removes artifacts for --base-name only:
  - <base-name>-hf/
  - <base-name>-f16.gguf
  - <base-name>-*.gguf

Options:
  --config-file <path>  Cleanup config txt (default: scripts/clean_up/clean_up_config.txt)
  --models-dir <path>   Models directory (default: ./models under repo root)
  --model-id <id>       Model id used for derived base-name default
  --base-name <name>    Base name prefix to clean (default: derived from --model-id)
  --all                 Remove all files/folders inside models-dir
  --dry-run             Show what would be removed without deleting
  --yes                 Skip confirmation prompt for destructive actions
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

MODELS_DIR="$(resolve_repo_path "${MODELS_DIR}")"

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "Models directory does not exist: ${MODELS_DIR}"
  exit 0
fi

confirm() {
  local prompt="$1"
  if [[ "${ASSUME_YES}" == "1" ]]; then
    return 0
  fi

  if [[ ! -t 0 ]]; then
    echo "Refusing destructive action in non-interactive shell without --yes." >&2
    return 1
  fi

  read -r -p "${prompt} [y/N]: " answer
  [[ "${answer}" =~ ^[Yy]$ ]]
}

remove_path() {
  local target="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] rm -rf ${target}"
  else
    rm -rf "${target}"
    echo "Removed: ${target}"
  fi
}

if [[ "${REMOVE_ALL}" == "1" ]]; then
  if ! confirm "This will remove EVERYTHING inside ${MODELS_DIR}. Continue?"; then
    echo "Aborted."
    exit 1
  fi

  shopt -s dotglob nullglob
  items=("${MODELS_DIR}"/*)
  shopt -u dotglob nullglob

  if [[ ${#items[@]} -eq 0 ]]; then
    echo "Nothing to remove in ${MODELS_DIR}."
    exit 0
  fi

  for item in "${items[@]}"; do
    remove_path "${item}"
  done

  echo "Cleanup complete: ${MODELS_DIR}"
  exit 0
fi

shopt -s nullglob
matches=(
  "${MODELS_DIR}/${BASE_NAME}-hf"
  "${MODELS_DIR}/${BASE_NAME}-f16.gguf"
  "${MODELS_DIR}/${BASE_NAME}-"*.gguf
)
shopt -u nullglob

declare -A seen=()
unique_matches=()
for path in "${matches[@]}"; do
  if [[ -n "${path}" && -z "${seen[${path}]:-}" ]]; then
    seen["${path}"]=1
    unique_matches+=("${path}")
  fi
done
matches=("${unique_matches[@]}")

if [[ ${#matches[@]} -eq 0 ]]; then
  echo "No artifacts found for base name '${BASE_NAME}' in ${MODELS_DIR}."
  exit 0
fi

if ! confirm "Remove ${#matches[@]} artifact(s) for base '${BASE_NAME}' in ${MODELS_DIR}?"; then
  echo "Aborted."
  exit 1
fi

for path in "${matches[@]}"; do
  remove_path "${path}"
done

echo "Cleanup complete for base '${BASE_NAME}'."
