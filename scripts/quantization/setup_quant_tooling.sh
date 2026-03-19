#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="CourseQARAG"
LLAMA_CPP_DIR="./llama.cpp"
USE_CUDA="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --llama-dir)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --cuda)
      USE_CUDA="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'HELP'
Usage: setup_quant_tooling.sh [--env ENV_NAME] [--llama-dir PATH] [--cuda auto|on|off]

What it does:
1) Installs Hugging Face CLI + conversion dependencies into the specified conda env.
2) Clones (or updates) llama.cpp under --llama-dir.
3) Builds llama.cpp binaries required by quantization scripts.

Examples:
  ./scripts/quantization/setup_quant_tooling.sh
  ./scripts/quantization/setup_quant_tooling.sh --env CourseQARAG --cuda on
  ./scripts/quantization/setup_quant_tooling.sh --llama-dir /opt/llama.cpp --cuda off
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Error: conda env '$ENV_NAME' does not exist." >&2
  exit 1
fi

echo "[1/4] Installing Hugging Face CLI + Python deps in env: $ENV_NAME"
conda run -n "$ENV_NAME" pip install --upgrade "huggingface_hub[cli]" sentencepiece

echo "[2/4] Preparing llama.cpp at: $LLAMA_CPP_DIR"
if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
  git -C "$LLAMA_CPP_DIR" pull --ff-only
else
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

echo "[3/4] Building llama.cpp"
pushd "$LLAMA_CPP_DIR" >/dev/null

if [[ "$USE_CUDA" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    USE_CUDA="on"
  else
    USE_CUDA="off"
  fi
fi

if [[ "$USE_CUDA" == "on" ]]; then
  echo "Building with CUDA enabled"
  cmake -S . -B build -DGGML_CUDA=ON
else
  echo "Building without CUDA"
  cmake -S . -B build
fi

cmake --build build -j"$(nproc)"
popd >/dev/null

echo "[4/4] Verifying tools"
[[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]] || { echo "Missing convert_hf_to_gguf.py" >&2; exit 1; }
[[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]] || { echo "Missing build/bin/llama-quantize" >&2; exit 1; }
[[ -x "$LLAMA_CPP_DIR/build/bin/llama-cli" ]] || { echo "Missing build/bin/llama-cli" >&2; exit 1; }

echo "Setup complete."
echo "- HF CLI env: $ENV_NAME"
echo "- llama.cpp dir: $LLAMA_CPP_DIR"
