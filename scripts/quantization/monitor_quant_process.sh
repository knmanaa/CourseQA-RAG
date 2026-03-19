#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/llama.cpp}"
BASE_NAME="${BASE_NAME:-qwen3_5-9b}"
QUANTS="${QUANTS:-q8_0 q6_K q5_K_M q4_K_M q4_0}"
INTERVAL="${INTERVAL:-10}"
WATCH_MODE="1"
RUN_PIPELINE="0"
ENV_NAME="${ENV_NAME:-CourseQARAG}"
NO_CLEAR="0"
RUN_PID=""
RUN_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --once)
      WATCH_MODE="0"
      shift
      ;;
    --run)
      RUN_PIPELINE="1"
      shift
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --no-clear)
      NO_CLEAR="1"
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: monitor_quant_process.sh [options]

Monitors the full quantization workflow status:
- Download (HF snapshot)
- HF -> GGUF conversion
- Quantization progress
- Validation activity

Options:
  --model-id <id>       Model id (default: Qwen/Qwen3.5-9B)
  --models-dir <path>   Models directory (default: ./models)
  --llama-dir <path>    llama.cpp directory (default: ./llama.cpp)
  --base-name <name>    Base model name (default: qwen3_5-9b)
  --quants "..."        Space-separated quant list to track
  --interval <sec>      Refresh interval in watch mode (default: 10)
  --once                Print one snapshot and exit
  --run                 Start run_all_quant.sh, then monitor in same terminal
  --env <name>          Conda env used with --run (default: CourseQARAG)
  --no-clear            Do not clear terminal between refreshes
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

HF_LOCAL_DIR="${MODELS_DIR}/${BASE_NAME}-hf"
F16_GGUF="${MODELS_DIR}/${BASE_NAME}-f16.gguf"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

human_size() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    echo "-"
  fi
}

is_running() {
  local pattern="$1"
  pgrep -af "$pattern" >/dev/null 2>&1
}

start_pipeline() {
  local run_script="${SCRIPT_DIR}/run_all_quant.sh"
  if [[ ! -x "${run_script}" ]]; then
    echo "Error: ${run_script} is missing or not executable." >&2
    exit 1
  fi

  if ! have_cmd conda; then
    echo "Error: conda not found in PATH (required for --run)." >&2
    exit 1
  fi

  RUN_LOG="${REPO_ROOT}/logs/quant_run_${BASE_NAME}_$(date '+%Y%m%d_%H%M%S').log"
  mkdir -p "${REPO_ROOT}/logs"

  conda run -n "${ENV_NAME}" "${run_script}" \
    --model-id "${MODEL_ID}" \
    --models-dir "${MODELS_DIR}" \
    --llama-dir "${LLAMA_CPP_DIR}" \
    --base-name "${BASE_NAME}" \
    --quants "${QUANTS}" \
    >"${RUN_LOG}" 2>&1 &

  RUN_PID="$!"
}

quant_done_count() {
  local done=0
  local total=0
  for quant in ${QUANTS}; do
    total=$((total + 1))
    if [[ -f "${MODELS_DIR}/${BASE_NAME}-${quant}.gguf" ]]; then
      done=$((done + 1))
    fi
  done
  echo "${done}/${total}"
}

current_stage() {
  if is_running "hf download ${MODEL_ID}" || is_running "huggingface-cli download ${MODEL_ID}"; then
    echo "Downloading model"
    return
  fi

  if is_running "convert_hf_to_gguf.py"; then
    echo "Converting HF -> GGUF"
    return
  fi

  if is_running "llama-quantize"; then
    echo "Quantizing"
    return
  fi

  if is_running "validate_quants.sh" || is_running "llama-cli -m ${MODELS_DIR}/${BASE_NAME}-"; then
    echo "Validating"
    return
  fi

  local q_count
  q_count="$(quant_done_count)"
  if [[ -f "${F16_GGUF}" && "${q_count}" == "$(awk '{print NF"/"NF}' <<<"${QUANTS}")" ]]; then
    echo "Completed"
    return
  fi

  if [[ -f "${F16_GGUF}" ]]; then
    echo "Idle (ready to quantize)"
    return
  fi

  if [[ -d "${HF_LOCAL_DIR}" ]]; then
    echo "Idle (download present, waiting conversion)"
    return
  fi

  echo "Not started"
}

print_status() {
  local now stage q_progress hf_size f16_size
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  stage="$(current_stage)"
  q_progress="$(quant_done_count)"
  hf_size="$(human_size "${HF_LOCAL_DIR}")"
  f16_size="$(human_size "${F16_GGUF}")"

  echo "[$now] Quantization Monitor"
  echo "- Model ID:            ${MODEL_ID}"
  echo "- Models dir:          ${MODELS_DIR}"
  echo "- Base name:           ${BASE_NAME}"
  echo "- Current stage:       ${stage}"
  echo "- HF snapshot dir:     ${HF_LOCAL_DIR} ($( [[ -d "${HF_LOCAL_DIR}" ]] && echo present || echo missing ), size: ${hf_size})"
  echo "- F16 GGUF:            ${F16_GGUF} ($( [[ -f "${F16_GGUF}" ]] && echo present || echo missing ), size: ${f16_size})"
  echo "- Quant progress:      ${q_progress}"

  echo "- Quant artifacts:"
  for quant in ${QUANTS}; do
    local path="${MODELS_DIR}/${BASE_NAME}-${quant}.gguf"
    if [[ -f "${path}" ]]; then
      echo "  - ${quant}: done ($(human_size "${path}"))"
    else
      echo "  - ${quant}: pending"
    fi
  done

  echo "- Process checks:"
  echo "  - hf/huggingface-cli download: $(is_running "(hf|huggingface-cli) download" && echo running || echo idle)"
  echo "  - convert_hf_to_gguf.py:       $(is_running "convert_hf_to_gguf.py" && echo running || echo idle)"
  echo "  - llama-quantize:              $(is_running "llama-quantize" && echo running || echo idle)"
  echo "  - llama-cli (validation):      $(is_running "llama-cli -m ${MODELS_DIR}/${BASE_NAME}-" && echo running || echo idle)"

  if [[ -n "${RUN_PID}" ]]; then
    if kill -0 "${RUN_PID}" >/dev/null 2>&1; then
      echo "- Wrapper run status:   running (pid ${RUN_PID})"
    else
      echo "- Wrapper run status:   finished (pid ${RUN_PID})"
    fi
    echo "- Wrapper log file:     ${RUN_LOG}"
    echo "- Tail logs with:       tail -f ${RUN_LOG}"
  fi
}

if [[ "${WATCH_MODE}" == "0" ]]; then
  if [[ "${RUN_PIPELINE}" == "1" ]]; then
    start_pipeline
  fi
  print_status
  exit 0
fi

if ! have_cmd pgrep; then
  echo "Error: pgrep is required for process monitoring." >&2
  exit 1
fi

if [[ "${RUN_PIPELINE}" == "1" ]]; then
  start_pipeline
fi

while true; do
  if [[ "${NO_CLEAR}" != "1" ]]; then
    clear || true
  fi
  print_status
  echo
  echo "Refreshing every ${INTERVAL}s. Press Ctrl+C to stop."
  sleep "${INTERVAL}"
done
