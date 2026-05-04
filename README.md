# CourseQA-RAG

Offline, privacy-preserving Teaching Assistant powered by RAG (Retrieval-Augmented Generation) using Haystack 2.x, FAISS, and a locally-served GGUF model.

## Quick Setup
First install conda and execute the followings:
```
conda env create -f conda_env_setup.yml
conda activate CourseQARAG
```
Next,

for Windows environment please execute `./scripts/setup_conda_env.ps1`;

> Note: For windows users, you will need to install Microsoft Visual Studio Build Tools 2022 as prompted in order to build llama-cpp-python wheels successfully.

for Linux environment please execute `./scripts/setup_conda_env.sh`;

The above process should work and you may skip the following details.

Finally, ensure you are in project root and conda env, then run:
```
python -m src.indexing
python -m src.retrieval_response
```

## Models

The app uses GGUF models from the HuggingFace repo [`knmanaa/courseQA`](https://huggingface.co/knmanaa/courseQA).

| Model file | Size | Default? |
|---|---|---|
| `qwen3_5-4b-q4_K_M.gguf` | ~2.6 GB | ✅ Yes |
| `qwen3_5-9b-q4_K_M.gguf` | ~5.8 GB | No |

On first run, the **4B model** is downloaded automatically to `models/` if it is not already present.

### Using the 9B model

To switch to the 9B model, download it manually from HuggingFace and place it in the `models/` folder:

```bash
# Option A – huggingface-cli (recommended)
huggingface-cli download knmanaa/courseQA qwen3_5-9b-q4_K_M.gguf --local-dir models

# Option B – direct wget / curl
curl -L -o models/qwen3_5-9b-q4_K_M.gguf \
  "https://huggingface.co/knmanaa/courseQA/resolve/main/qwen3_5-9b-q4_K_M.gguf"
```

Then tell the app to use it by setting the environment variable before running:

```bash
export COURSEQA_MODEL_FILE=qwen3_5-9b-q4_K_M.gguf
python -m src.retrieval_response
```

> **Note:** The 9B model requires ~6 GB of VRAM for full GPU offload. On an 8 GB card (e.g. RTX 3070) you may need to reduce GPU layers: `export COURSEQA_N_GPU_LAYERS=30`

### Embedding model

The `sentence-transformers/all-MiniLM-L6-v2` embedding model (~22 MB) is also downloaded automatically to `deploy/embeddings/all-MiniLM-L6-v2/` on first run of either `src.indexing` or `src.retrieval_response`.

---

## GPU Acceleration (Optional but Recommended)

The bootstrap script detects whether CUDA is available and installs the matching PyTorch wheel plus a CUDA-enabled `llama-cpp-python` build when possible. If no GPU is detected, it keeps the CPU-only stack.

By default the pipeline automatically uses your GPU for both embeddings and LLM inference if it is properly configured.

If you want to rerun just the environment bootstrap later:
```bash
bash scripts/setup_conda_env.sh CourseQARAG
```

**Install PyTorch with CUDA manually (if needed):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Rebuild llama-cpp-python for GPU manually (if needed):**
```cmd
set CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

## Quantization Script Guide

See `scripts/quantization/README.md` for:
- what each quantization script does
- one-command wrapper usage
- examples with and without monitoring


## Folder Structure

```
CourseQA-RAG/
├── src/                        # Pipeline source code
│   ├── ingest.py               # PDF extraction → Haystack Documents
│   ├── store.py                # FAISSDocumentStore, embedder, and retriever factory functions
│   ├── indexer.py              # Haystack indexing pipeline (cleaner → splitter → embedder → writer)
│   ├── nodes.py                # Custom @component classes: QueryRouter, DocumentGrader,
│   │                           #   AnswerBuilder, HallucinationGrader, AnswerGrader,
│   │                           #   QueryReformulator, OutOfScopeResponder
│   ├── pipeline.py             # Haystack Pipeline wiring (add_component / connect)
│   ├── cache.py                # Embedding cache (content-hash → vector) + TTL query cache
│   └── app.py                  # Gradio UI with Chat, Documents, Monitor, and Settings tabs
│
├── eval/                       # Evaluation scripts and dataset
│   ├── eval_dataset.json       # 50–100 hand-crafted QA pairs with ground-truth answers,
│   │                           #   source doc/page, and category labels
│   ├── run_retrieval_eval.py   # Benchmarks Precision@K, Recall@K, MRR across chunking
│   │                           #   strategies, chunk sizes, K values, and index types
│   ├── run_generation_eval.py  # Measures answer accuracy, relevance, and factual consistency
│   └── run_models_bench.py     # Records TTFT, total generation time, peak RAM, and tok/s
│                               #   across quant levels on both DGX Spark and laptop
│
├── data/
│   ├── raw/                    # Original PDF course materials
│   │   ├── lectures/           # Lecture slide PDFs
│   │   ├── textbook/           # Textbook chapter PDFs
│   │   └── assignments/        # Assignment / worksheet PDFs
│   ├── processed/              # Per-document extracted text as JSON
│   │                           #   Format: { "text": ..., "pages": [...], "metadata": {...} }
│   ├── faiss_store/            # FAISSDocumentStore persistence
│   │                           #   faiss_document_store.db — SQLite chunk text + metadata
│   │                           #   faiss_index.faiss       — FAISS binary index file
│   └── embeddings_cache/       # Pre-computed embedding cache for incremental re-indexing
│                               #   cache.npy        — stacked embedding vectors
│                               #   cache_index.json — content-hash → row mapping
│
├── models/                     # GGUF model files (auto-downloaded on first run)
│                               #   qwen3_5-4b-q4_K_M.gguf  — default 4B model (~2.6 GB)
│                               #   qwen3_5-9b-q4_K_M.gguf  — optional 9B model (~5.8 GB, manual download)
│
├── scripts/
│   └── quantization/           # Shell scripts for model conversion and quantization
│                               #   convert_to_gguf.sh  — downloads model from HF and converts HF → GGUF (FP16)
│                               #   quantize_all.sh     — loops over quant levels (q8_0, q5_K_M, q4_K_M, q4_0)
│                               #   validate_quants.sh  — spot-checks each GGUF with a test prompt
│                               #   Run these on DGX Spark (requires llama.cpp built with GGML_CUDA=1)
│
├── deploy/                     # Self-contained deployment bundle for consumer laptops
│   ├── models/                 # Q4_K_M GGUF file for deployment (~4.9 GB)
│   ├── embeddings/
│   │   └── all-MiniLM-L6-v2/  # sentence-transformers weights (~22 MB)
│   ├── faiss_store/            # Copied faiss_index.faiss + faiss_document_store.db
│   ├── embeddings_cache/       # Copied cache.npy + cache_index.json
│   └── src/                    # Copied pipeline source code
│
├── requirements.txt
└── README.md
```
