# Quantization Scripts Guide

This folder contains shell scripts to prepare and quantize `Qwen/Qwen3.5-9B` into GGUF variants.

## What each script does

- `setup_quant_tooling.sh`
  - Installs Hugging Face CLI support in your conda env.
  - Clones or updates `llama.cpp`.
  - Builds required binaries (`llama-quantize`, `llama-cli`) and verifies tooling.

- `convert_to_gguf.sh`
  - Downloads the HF model snapshot.
  - Converts HF weights to FP16 GGUF (`*-f16.gguf`).
  - Supports both `hf` and `huggingface-cli` commands.

- `quantize_all.sh`
  - Quantizes FP16 GGUF into configured quant formats.
  - Default quants: `q8_0 q6_K q5_K_M q4_K_M q4_0`.

- `validate_quants.sh`
  - Runs a prompt test against each generated GGUF file.
  - Useful for quick sanity checks of output quality and runtime stability.

- `monitor_quant_process.sh`
  - Prints live status snapshots of the full process (download/convert/quantize/validate).
  - Can also launch the process with `--run` and monitor from the same terminal.

- `run_all_quant.sh` (main wrapper)
  - End-to-end orchestrator for:
    1. Convert HF → FP16 GGUF
    2. Quantize all configured levels
  - Validation is run separately via `validate_quants.sh`.
  - Optional flags include fresh cleanup and conda env execution.

## Wrapper script examples

Make sure your terminal is already at the project root. Run full process with **live terminal output**:

```bash
conda run --no-capture-output -n CourseQARAG ./scripts/quantization/run_all_quant.sh --fresh
```

The scripts automatically read defaults from `scripts/quantization/model_config.txt`.

Run validation separately:

```bash
conda run --no-capture-output -n CourseQARAG ./scripts/quantization/validate_quants.sh
```

Clean model artifacts (new script):

```bash
./scripts/clean_up/clean_models.sh --yes
```

Cleanup defaults are auto-read from `scripts/clean_up/clean_up_config.txt`.

Clean all files under `models/`:

```bash
./scripts/clean_up/clean_models.sh --all --yes
```

## Download and quantize a specific Hugging Face model

Use `MODEL_ID` to target any HF model repo. `BASE_NAME` is optional; if omitted, it is auto-derived from `MODEL_ID`.

Recommended: set values in `scripts/quantization/model_config.txt` once, then run scripts normally.

Example `model_config.txt` entries:

```txt
MODEL_ID=Qwen/Qwen3.5-9B
PROMPT=Explain gradient descent in 3 concise bullet points.
N_PREDICT=128
TEMP=0.1
```

For gated/private models, add your token:

```txt
HF_TOKEN=<your_huggingface_token>
```

Example (Qwen3.5):

```bash
conda run --no-capture-output -n CourseQARAG \
  MODEL_ID="Qwen/Qwen3.5-9B" \
  ./scripts/quantization/run_all_quant.sh --fresh
```

Example (custom quants and explicit base name):

```bash
conda run --no-capture-output -n CourseQARAG \
  MODEL_ID="Qwen/Qwen3.5-9B" \
  BASE_NAME="qwen3_5-9b" \
  ./scripts/quantization/run_all_quant.sh --fresh --quants "q8_0 q6_K q5_K_M"
```

Then run validation:

```bash
conda run --no-capture-output -n CourseQARAG \
  MODEL_ID="Qwen/Qwen3.5-9B" \
  ./scripts/quantization/validate_quants.sh
```

Validation can also read all settings from config txt (no extra env needed):

```bash
conda run --no-capture-output -n CourseQARAG ./scripts/quantization/validate_quants.sh
```

## Gated model example: google/gemma-3-27b-it

1) Ensure your Hugging Face account has accepted access for `google/gemma-3-27b-it`.

2) Edit `scripts/quantization/model_config.txt`:

```txt
MODEL_ID=google/gemma-3-27b-it
HF_TOKEN=<your_huggingface_token>
QUANTS=q8_0 q6_K q5_K_M q4_K_M q4_0
PROMPT=Explain gradient descent in 3 concise bullet points.
N_PREDICT=128
TEMP=0.1
```

3) Run quantization:

```bash
conda run --no-capture-output -n CourseQARAG ./scripts/quantization/run_all_quant.sh --fresh
```

Optional CLI override for token:

```bash
conda run --no-capture-output -n CourseQARAG \
  ./scripts/quantization/run_all_quant.sh --fresh --model-id google/gemma-3-27b-it --hf-token "${HF_TOKEN}"
```

## Notes

- Artifacts are preserved by default. `run_all_quant.sh` does not delete existing files unless you pass `--fresh`.
- `--fresh` removes prior artifacts for the selected `--base-name` before starting (if matching artifacts exist).
- `MODEL_ID` is supported across convert/quantize/validate/cleanup scripts; `BASE_NAME` defaults to a sanitized name derived from `MODEL_ID` unless you explicitly set `BASE_NAME`.
- `run_all_quant.sh` and `validate_quants.sh` read `scripts/quantization/model_config.txt` automatically.
- `clean_models.sh` reads `scripts/clean_up/clean_up_config.txt` automatically.
- Priority order is: CLI args > environment variables > `model_config.txt` > script defaults.
- Cleanup priority order is: CLI args > environment variables > `clean_up_config.txt` > script defaults.
- Dry-run check example: `--model-id meta-llama/Llama-3.1-8B-Instruct` derives base name `llama-3_1-8b-instruct`.
- You can customize model/base name/quants via `run_all_quant.sh --help`.
- You can customize validation prompt/temperature/tokens via config txt, env vars, or CLI flags on `validate_quants.sh`.
- Cleanup options are available via `./scripts/clean_up/clean_models.sh --help`.
