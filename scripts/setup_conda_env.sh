#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-CourseQARAG}"
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-0.3.20}"
PYTHON_EXE="/shared/conda_envs/${ENV_NAME}/bin/python3"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Error: conda env '$ENV_NAME' does not exist." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Error: Python executable not found at $PYTHON_EXE" >&2
  exit 1
fi

get_cuda_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_line
    cuda_line="$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' '/CUDA Version:/ {print $2; exit}')"
    if [[ -n "$cuda_line" ]]; then
      echo "$cuda_line" | awk '{print $1}'
      return 0
    fi
  fi

  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>/dev/null | awk -F', ' '/release/ {print $2; exit}' | sed 's/release //'
    return 0
  fi

  echo ""
}

pick_torch_tag() {
  local cuda_version="$1"
  local major minor
  major="${cuda_version%%.*}"
  minor="${cuda_version#*.}"
  minor="${minor%%.*}"

  if [[ -z "$major" || -z "$minor" ]]; then
    echo "cpu"
    return 0
  fi

  if [[ "$major" -ge 13 ]]; then
    echo "cu128"
  elif [[ "$major" -eq 12 && "$minor" -ge 8 ]]; then
    echo "cu128"
  elif [[ "$major" -eq 12 && "$minor" -ge 6 ]]; then
    echo "cu126"
  elif [[ "$major" -eq 12 && "$minor" -ge 4 ]]; then
    echo "cu124"
  elif [[ "$major" -eq 12 && "$minor" -ge 1 ]]; then
    echo "cu121"
  elif [[ "$major" -eq 11 && "$minor" -ge 8 ]]; then
    echo "cu118"
  else
    echo "cpu"
  fi
}

CUDA_VERSION="$(get_cuda_version)"
if [[ -n "$CUDA_VERSION" ]]; then
  TORCH_TAG="$(pick_torch_tag "$CUDA_VERSION")"
  echo "Detected CUDA version: $CUDA_VERSION"
  echo "Selected PyTorch wheel tag: $TORCH_TAG"
else
  TORCH_TAG="cpu"
  echo "No CUDA runtime detected. Installing CPU-only PyTorch and llama-cpp-python."
fi

echo "[1/4] Upgrading pip tooling in $ENV_NAME"
"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

echo "[2/4] Installing PyTorch"
if [[ "$TORCH_TAG" == "cpu" ]]; then
  "$PYTHON_EXE" -m pip install --upgrade torch torchvision torchaudio
else
  "$PYTHON_EXE" -m pip install --upgrade torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_TAG}"
fi

echo "[3/4] Installing llama-cpp-python"
"$PYTHON_EXE" -m pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true
if [[ "$TORCH_TAG" == "cpu" ]]; then
  "$PYTHON_EXE" -m pip install --upgrade "llama-cpp-python==${LLAMA_CPP_VERSION}"
else
  CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 "$PYTHON_EXE" -m pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==${LLAMA_CPP_VERSION}"
fi

echo "[4/4] Verifying installs"
"$PYTHON_EXE" - <<'PY'
import torch
import llama_cpp

print('torch.cuda.is_available =', torch.cuda.is_available())
print('torch.version.cuda =', getattr(torch.version, 'cuda', None))
print('llama_cpp version =', getattr(llama_cpp, '__version__', 'unknown'))
fn = getattr(llama_cpp.llama_cpp, 'llama_supports_gpu_offload', None)
print('llama_supports_gpu_offload =', bool(fn()) if callable(fn) else '<not exposed>')
PY

echo "Setup complete for env: $ENV_NAME"
