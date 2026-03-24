# CourseQA-RAG

Offline, privacy-preserving Teaching Assistant powered by RAG (Retrieval-Augmented Generation) using Haystack 2.x, FAISS, and a locally-served GGUF model.

## Conda Env Setup
Install conda, and execute following:
```
conda env create -f conda_env_setup.yml
conda activate CourseQARAG
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
├── models/                     # GGUF model files (generated on DGX Spark)
│                               #   deepseek-8b-f16.gguf, deepseek-8b-q8_0.gguf,
│                               #   deepseek-8b-q5_K_M.gguf, deepseek-8b-q4_K_M.gguf, etc.
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
