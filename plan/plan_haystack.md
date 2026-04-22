# Project 10: On-Device RAG for Course QA — Haystack Edition

## Overview

Build a privacy-preserving, fully offline Teaching Assistant using Retrieval-Augmented Generation (RAG) to answer student questions from course materials. Use **DGX Spark (GB10)** for GPU-intensive preparation (model conversion, bulk embedding, benchmarking) and deploy the final pipeline on **mid-range consumer laptops (16 GB RAM, CPU or low-end GPU)**.

**Core library stack:**

| Library | Role |
|---|---|
| **`haystack-ai`** | Haystack 2.x core — `Pipeline`, `DocumentSplitter`, `DocumentCleaner`, `SentenceTransformersDocumentEmbedder` / `SentenceTransformersTextEmbedder`, `PromptBuilder`, `@component` decorator |
| **`faiss-haystack`** | Haystack 2.x FAISS integration — `FAISSDocumentStore` + `FAISSEmbeddingRetriever` (backed by `faiss-cpu`) |
| **`sentence-transformers`** | Embedding model (`all-MiniLM-L6-v2`, 384-dim, 22 MB) used inside Haystack's built-in `SentenceTransformers*Embedder` components |

**Key design decisions:**

- **Haystack 2.x pipelines replace LangChain / LangGraph** — Haystack 2.x provides a component-based `Pipeline` with `add_component()` / `connect()` wiring, the `@component` decorator for custom logic, and built-in components for splitting, embedding, retrieval, prompting, and generation.
- **`FAISSDocumentStore`** — provided by `faiss-haystack` integration; supports `Flat`, `HNSW`, and `IVF` index types out of the box via `faiss_index_factory_str`, no raw FAISS code needed.
- **Single quantization path (GGUF via llama.cpp)** — portable artifacts that run on any hardware.
- **DeepSeek-R1-Distill-Llama-8B** as the generation model (8 B params, strong reasoning).
- **`all-MiniLM-L6-v2`** as the embedding model via `sentence-transformers`.
- **Gradio** for the demo UI.

**Hardware split:**

| Task | DGX Spark (GB10) | Consumer Laptop (16 GB) |
|---|---|---|
| GGUF model conversion (FP16 → Q4_K_M etc.) | ✅ | — |
| Bulk PDF embedding (50+ PDFs) | ✅ (fast) | ✅ (slower, still works) |
| FAISS index construction | ✅ | ✅ |
| Benchmarking all quant levels | ✅ | ✅ (subset) |
| Final RAG pipeline demo | — | ✅ |

---

## Timeline (6 Weeks)

| Week | Phase | Deliverables |
|---|---|---|
| 1 | Environment + Data Prep | Dev environments on both machines, PDF ingestion via Haystack `DocumentCleaner` + `DocumentSplitter`, clean text corpus |
| 1–2 | Chunking + Embedding | Haystack `DocumentSplitter` strategies, `SentenceTransformersDocumentEmbedder`, `FAISSDocumentStore` |
| 2–3 | Model Quantization | GGUF quants (FP16, Q8_0, Q5_K_M, Q4_K_M, Q4_0), validation |
| 3–4 | RAG Pipeline | Custom Haystack 2.x pipeline with `@component` grading/hallucination components, llama.cpp server |
| 4 | UI + Demo | Gradio chat interface with document management + monitoring |
| 4–5 | Evaluation | Retrieval metrics, generation quality, system benchmarks |
| 5–6 | Optimization + Report | Measured optimization gains, final report, demo prep |

---

## Phase 1 — Environment & Data Preparation (Week 1)

### Step 1: Set up DGX Spark environment

1. Install NVIDIA AI Workbench on DGX Spark.
2. Verify Docker GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
   ```
3. Obtain HuggingFace token (`HF_TOKEN`) for model download. No NVIDIA API key needed (everything runs locally).
4. Install core tools:
   ```bash
   pip install haystack-ai faiss-haystack sentence-transformers
   pip install pymupdf gradio pydantic tqdm psutil cachetools
   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j$(nproc) GGML_CUDA=1
   ```

### Step 2: Set up consumer laptop environment

```bash
conda create -n rag python=3.11 -y && conda activate rag
pip install haystack-ai faiss-haystack sentence-transformers
pip install llama-cpp-python pymupdf gradio pydantic tqdm psutil cachetools
```

Verify installations:

```bash
python -c "from haystack_integrations.document_stores.faiss import FAISSDocumentStore; print('Haystack + FAISS OK')"
python -c "from haystack.components.embedders import SentenceTransformersTextEmbedder; print('Embedder OK')"
python -c "from llama_cpp import Llama; print('llama-cpp OK')"
```

### Step 3: Collect and organize course materials

```
data/
├── raw/                        # Original PDFs
│   ├── lectures/
│   ├── textbook/
│   └── assignments/
├── processed/                  # Extracted text (per-doc JSON)
│   └── {doc_id}.json           # {"text": ..., "pages": [...], "metadata": {...}}
├── faiss_store/                # FAISSDocumentStore persistence
│   ├── faiss_document_store.db # SQLite metadata backend
│   └── faiss_index.faiss       # FAISS binary index
├── embeddings_cache/           # Precomputed embedding cache
│   ├── cache.npy
│   └── cache_index.json
└── manifest.json               # PDF metadata manifest
```

### Step 4: Build PDF ingestion pipeline

**File: `src/ingest.py`**

- Use `PyMuPDF` (fitz) for text extraction — handles text-heavy docs and slide-style PDFs.
- Fallback to `pytesseract` OCR for scanned pages (detect via low character count).
- Output: list of Haystack `Document` objects with `content` and `meta` fields.
- Deduplication via file-content hash in `manifest.json`.

```python
from haystack.dataclasses import Document
import fitz  # PyMuPDF

def extract_pdf(path: str) -> list[Document]:
    """Extract pages from a PDF and return Haystack Documents."""
    doc = fitz.open(path)
    documents = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if not text:
            continue
        documents.append(Document(
            content=text,
            meta={
                "source": path,
                "page": page_num,
                "total_pages": len(doc),
            }
        ))
    return documents
```

---

## Phase 2 — Chunking & Embedding (Week 1–2)

### Step 5: Implement chunking with Haystack `DocumentSplitter`

Haystack 2.x replaces the monolithic `PreProcessor` with composable components: `DocumentCleaner` (whitespace / header cleanup) and `DocumentSplitter` (chunking). Three configurations to benchmark:

**Strategy 1 — Fixed-size with overlap (baseline):**

```python
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
)

splitter_fixed = DocumentSplitter(
    split_by="word",
    split_length=200,          # ~200 words ≈ ~256 tokens
    split_overlap=20,
)
```

Benchmark three sizes: `split_length` in {200, 400, 800} words with proportional overlap.

**Strategy 2 — Sentence-based splitting:**

```python
splitter_sentence = DocumentSplitter(
    split_by="sentence",
    split_length=5,            # 5 sentences per chunk
    split_overlap=1,
)
```

Preserves sentence boundaries, good for conceptual coherence.

**Strategy 3 — Passage-based splitting:**

```python
splitter_passage = DocumentSplitter(
    split_by="passage",        # Splits on "\n\n"
    split_length=1,
    split_overlap=0,
)
```

Natural paragraph boundaries. Post-filter to merge very short passages and cap very long ones.

Each chunk inherits metadata from its parent `Document` automatically. The `DocumentSplitter` adds `split_id` and `split_idx_start` to each chunk’s meta.

### Step 6: Initialize `FAISSDocumentStore` and retriever / embedder components

**File: `src/store.py`**

```python
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

def create_store(index_type: str = "Flat", dims: int = 384) -> FAISSDocumentStore:
    """
    Create a FAISSDocumentStore (Haystack 2.x).
    index_type: "Flat" | "HNSW" | "IVF256,Flat" | "IVF256,SQ8"
    """
    return FAISSDocumentStore(
        sql_url="sqlite:///data/faiss_store/faiss_document_store.db",
        faiss_index_factory_str=index_type,
        embedding_dim=dims,
        return_embedding=True,
        similarity="cosine",
    )

def create_doc_embedder() -> SentenceTransformersDocumentEmbedder:
    """Embedder for indexing: embeds Document objects in batch."""
    return SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

def create_text_embedder() -> SentenceTransformersTextEmbedder:
    """Embedder for query time: embeds a single query string."""
    return SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

def create_retriever(document_store: FAISSDocumentStore, top_k: int = 5) -> FAISSEmbeddingRetriever:
    """Retriever that searches FAISSDocumentStore by embedding similarity."""
    return FAISSEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k,
    )
```

**Index types to benchmark** (via `faiss_index_factory_str`):

| Haystack Factory String | FAISS Index | Search | Memory | Use |
|---|---|---|---|---|
| `"Flat"` | `IndexFlatIP` | O(N), exact | Full | Baseline |
| `"HNSW"` | `IndexHNSWFlat` | O(log N), approx | Full + graph | **Primary deployment** |
| `"IVF256,Flat"` | `IndexIVFFlat` | O(N/nprobe) | Full + centroids | Alternative |
| `"IVF256,SQ8"` | `IndexIVFScalarQuantizer` | O(N/nprobe) | ~1/4 | Memory optimization |

### Step 7: Write documents and update embeddings

**File: `src/indexer.py`**

In Haystack 2.x, indexing is a pipeline itself: `DocumentCleaner → DocumentSplitter → SentenceTransformersDocumentEmbedder → DocumentWriter`.

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from src.ingest import extract_pdf
from pathlib import Path

def build_index(pdf_dir: str, index_type: str = "HNSW"):
    store = FAISSDocumentStore(
        sql_url="sqlite:///data/faiss_store/faiss_document_store.db",
        faiss_index_factory_str=index_type,
        embedding_dim=384,
        return_embedding=True,
        similarity="cosine",
    )

    # Build indexing pipeline
    indexing = Pipeline()
    indexing.add_component("cleaner",    DocumentCleaner(remove_empty_lines=True,
                                                         remove_extra_whitespaces=True))
    indexing.add_component("splitter",   DocumentSplitter(split_by="word",
                                                          split_length=200,
                                                          split_overlap=20))
    indexing.add_component("embedder",   SentenceTransformersDocumentEmbedder(
                                             model="sentence-transformers/all-MiniLM-L6-v2"))
    indexing.add_component("writer",     DocumentWriter(document_store=store))

    indexing.connect("cleaner",   "splitter")
    indexing.connect("splitter",  "embedder")
    indexing.connect("embedder",  "writer")

    # Ingest all PDFs
    pdf_paths = list(Path(pdf_dir).rglob("*.pdf"))
    all_docs = []
    for pdf in pdf_paths:
        all_docs.extend(extract_pdf(str(pdf)))

    indexing.run({"cleaner": {"documents": all_docs}})

    # Persist to disk
    store.save("data/faiss_store/faiss_index.faiss")
    print(f"Indexed documents from {len(pdf_paths)} PDFs.")
```

**Embedding cache logic** (optimization #3):

```python
import hashlib, json, numpy as np

CACHE_PATH = "data/embeddings_cache/cache.npy"
INDEX_PATH = "data/embeddings_cache/cache_index.json"

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def load_cache():
    try:
        vecs = np.load(CACHE_PATH)
        with open(INDEX_PATH) as f:
            idx = json.load(f)
        return vecs, idx
    except FileNotFoundError:
        return np.empty((0, 384), dtype=np.float32), {}

def save_cache(vecs, idx):
    np.save(CACHE_PATH, vecs)
    with open(INDEX_PATH, "w") as f:
        json.dump(idx, f)
```

Integrate cache into the indexing pipeline: before running `SentenceTransformersDocumentEmbedder`, check if a document’s `content_hash` already exists in cache. If so, set `doc.embedding` directly and bypass the embedder for that document. Write all documents (with embeddings) into `FAISSDocumentStore` via `DocumentWriter`.

---

## Phase 3 — Model Quantization on DGX Spark (Week 2–3)

### Step 8: Convert to GGUF and quantize

All scripts live in **`scripts/quantization/`** and are intended to be run on DGX Spark (llama.cpp must be built with `GGML_CUDA=1`).

**`scripts/quantization/convert_to_gguf.sh`** — downloads the model from HuggingFace and converts it to GGUF FP16 baseline:

```bash
# 1. Download model
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --local-dir ./models/deepseek-8b-hf

# 2. Convert HF → GGUF (FP16 baseline)
python llama.cpp/convert_hf_to_gguf.py ./models/deepseek-8b-hf \
    --outfile ./models/deepseek-8b-f16.gguf --outtype f16
```

**`scripts/quantization/quantize_all.sh`** — loops over all quant levels and produces one GGUF per level:

```bash
# 3. Quantize to multiple levels
for QUANT in q8_0 q5_K_M q4_K_M q4_0; do
    ./llama.cpp/llama-quantize ./models/deepseek-8b-f16.gguf \
        ./models/deepseek-8b-${QUANT}.gguf ${QUANT}
done
```

**`scripts/quantization/validate_quants.sh`** — spot-checks each GGUF with a test prompt (see Step 9).

Expected output sizes:

| Quant | File Size | RAM at Inference (~4 K ctx) | Quality |
|---|---|---|---|
| F16 | ~16 GB | ~17 GB | Baseline (best) |
| Q8_0 | ~8.5 GB | ~9.5 GB | Near-lossless |
| Q5_K_M | ~5.5 GB | ~6.5 GB | Very good |
| **Q4_K_M** | **~4.9 GB** | **~6 GB** | **Good — primary deploy target** |
| Q4_0 | ~4.3 GB | ~5.5 GB | Acceptable, fastest |

### Step 9: Validate quantized models

Run **`scripts/quantization/validate_quants.sh`** — iterates over every GGUF in `models/` and fires a test prompt:

```bash
for GGUF in ./models/deepseek-8b-*.gguf; do
    echo "=== $GGUF ==="
    ./llama.cpp/llama-cli -m "$GGUF" \
        -p "What is gradient descent?" -n 128 --temp 0.1
done
```

Verify: coherent output, no garbage tokens, reasonable latency.

### Step 10: Package deployment artifacts

```
deploy/
├── models/
│   └── deepseek-8b-q4_K_M.gguf          # ~4.9 GB
├── embeddings/
│   └── all-MiniLM-L6-v2/                # ~22 MB (sentence-transformers weights)
├── faiss_store/
│   ├── faiss_index.faiss                 # Persisted FAISS index
│   └── faiss_document_store.db           # SQLite with chunk text + metadata
├── embeddings_cache/
│   ├── cache.npy
│   └── cache_index.json
└── src/                                  # Pipeline code
```

Total deployment size: **~5.5 GB**.

---

## Phase 4 — RAG Pipeline Implementation (Week 3–4)

### Step 11: Implement agentic RAG with custom Haystack 2.x pipeline

Haystack 2.x pipelines are directed graphs of **components** (decorated with `@component`). We implement the agentic pattern (Adaptive RAG + Corrective RAG + Self-RAG) using custom `@component`-decorated classes.

**Flow:**

```
User Query
    │
    ▼
┌──────────────┐
│  QueryRouter  │── "out_of_scope" ──► Polite refusal
│  (@component)  │
└──────┬────────┘
       │ "course_material"
       ▼
┌───────────────────────────┐
│ TextEmbedder +             │
│ FAISSEmbeddingRetriever    │  Embed query → FAISS Top-K
└──────┬─────────────────────┘
       ▼
┌──────────────────┐
│ DocumentGrader    │  LLM judges relevance per chunk
│ (@component)      │── all irrelevant ──► QueryReformulator → loop back
└──────┬────────────┘
       │ has relevant docs
       ▼
┌──────────────────┐
│ AnswerBuilder     │  Build prompt from graded context → generate answer
│ (@component)      │
└──────┬────────────┘
       ▼
┌────────────────────────┐
│ HallucinationGrader     │── not grounded ──► re-generate (max 2 retries)
│ (@component)            │
└──────┬──────────────────┘
       │ grounded
       ▼
┌──────────────────┐
│ AnswerGrader      │── not useful ──► QueryReformulator → loop back
│ (@component)      │
└──────┬────────────┘
       │ useful
       ▼
   Return Answer + Sources
```

### Step 12: Implement custom Haystack 2.x components

**File: `src/nodes.py`**

```python
from haystack import component, Document
from typing import List, Optional
import requests

LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"

def llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
    """Call the local llama-cpp-python OpenAI-compatible server."""
    resp = requests.post(LLM_URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    return resp.json()["choices"][0]["message"]["content"]


@component
class QueryRouter:
    """Routes queries to course_material retrieval or out_of_scope refusal."""

    @component.output_types(
        course_query=Optional[str],
        out_of_scope_query=Optional[str],
    )
    def run(self, query: str):
        prompt = (
            "You are a query router for a course QA system.\n"
            "Decide if the following question is about course material or out of scope.\n"
            "Respond with ONLY 'course_material' or 'out_of_scope'.\n\n"
            f"Question: {query}"
        )
        decision = llm_call(prompt, max_tokens=20).strip().lower()
        if "out_of_scope" in decision:
            return {"out_of_scope_query": query}
        return {"course_query": query}


@component
class DocumentGrader:
    """Grades each retrieved document for relevance. Keeps only relevant ones."""

    @component.output_types(
        query=Optional[str],
        relevant_documents=Optional[List[Document]],
        irrelevant_query=Optional[str],
    )
    def run(self, query: str, documents: List[Document]):
        graded = []
        for doc in documents:
            prompt = (
                "You are a relevance grader.\n"
                "Does the following document contain information relevant to the question?\n"
                "Respond with ONLY 'yes' or 'no'.\n\n"
                f"Question: {query}\n\nDocument:\n{doc.content[:1000]}"
            )
            verdict = llm_call(prompt, max_tokens=10).strip().lower()
            if "yes" in verdict:
                graded.append(doc)

        if graded:
            return {"query": query, "relevant_documents": graded}
        else:
            return {"irrelevant_query": query}


@component
class AnswerBuilder:
    """Builds a prompt from query + graded documents, calls LLM, returns answer."""

    @component.output_types(
        query=str, answer=str,
        documents=List[Document], sources=list,
    )
    def run(self, query: str, documents: List[Document]):
        context = "\n\n---\n\n".join(
            f"[Source: {d.meta.get('source','?')}, Page {d.meta.get('page','?')}]\n{d.content}"
            for d in documents
        )
        prompt = (
            "You are a helpful teaching assistant. Answer the question using ONLY "
            "the provided context. If the context does not contain enough information, "
            "say so. Cite sources.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        answer = llm_call(prompt, max_tokens=512)
        sources = [
            {"source": d.meta.get("source"), "page": d.meta.get("page")}
            for d in documents
        ]
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "sources": sources,
        }


@component
class HallucinationGrader:
    """Checks if the answer is grounded in the provided documents."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._retry_count = 0

    @component.output_types(
        query=Optional[str], answer=Optional[str],
        documents=Optional[List[Document]], sources=Optional[list],
        regenerate_query=Optional[str],
        regenerate_documents=Optional[List[Document]],
    )
    def run(self, query: str, answer: str,
            documents: List[Document], sources: list = None):
        context = "\n".join(d.content[:500] for d in documents)
        prompt = (
            "You are a hallucination grader.\n"
            "Is the following answer fully supported by the provided context?\n"
            "Respond with ONLY 'yes' or 'no'.\n\n"
            f"Context:\n{context}\n\nAnswer:\n{answer}"
        )
        verdict = llm_call(prompt, max_tokens=10).strip().lower()
        if "yes" in verdict or self._retry_count >= self.max_retries:
            self._retry_count = 0
            return {"query": query, "answer": answer,
                    "documents": documents, "sources": sources or []}
        else:
            self._retry_count += 1
            return {"regenerate_query": query,
                    "regenerate_documents": documents}


@component
class AnswerGrader:
    """Checks if the answer actually addresses the question."""

    @component.output_types(
        answer=Optional[str], sources=Optional[list],
        reformulate_query=Optional[str],
    )
    def run(self, query: str, answer: str,
            documents: List[Document], sources: list = None):
        prompt = (
            "You are an answer grader.\n"
            "Does the following answer adequately address the question?\n"
            "Respond with ONLY 'yes' or 'no'.\n\n"
            f"Question: {query}\n\nAnswer:\n{answer}"
        )
        verdict = llm_call(prompt, max_tokens=10).strip().lower()
        if "yes" in verdict:
            return {"answer": answer, "sources": sources or []}
        else:
            return {"reformulate_query": query}


@component
class QueryReformulator:
    """Rewrites the query for a better retrieval attempt."""

    @component.output_types(query=str)
    def run(self, query: str):
        prompt = (
            "You are a query rewriter for a course QA system.\n"
            "The original query did not retrieve useful results.\n"
            "Rewrite the query to improve retrieval. Return ONLY the rewritten query.\n\n"
            f"Original query: {query}"
        )
        new_query = llm_call(prompt, max_tokens=100).strip()
        return {"query": new_query}


@component
class OutOfScopeResponder:
    """Returns a polite refusal for out-of-scope questions."""

    @component.output_types(answer=str, sources=list)
    def run(self, query: str):
        return {
            "answer": "I'm sorry, this question appears to be outside the scope "
                      "of the course material. Please ask a question related to "
                      "the course content.",
            "sources": [],
        }
```

### Step 13: Wire the Haystack 2.x pipeline

**File: `src/pipeline.py`**

```python
from haystack import Pipeline
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from src.nodes import (
    QueryRouter, DocumentGrader, AnswerBuilder,
    HallucinationGrader, AnswerGrader, QueryReformulator,
    OutOfScopeResponder,
)

def build_rag_pipeline(faiss_index_path: str, db_url: str) -> Pipeline:
    # Load persisted FAISS store
    store = FAISSDocumentStore.load(
        faiss_file_path=faiss_index_path,
        sql_url=db_url,
    )

    # Build pipeline with add_component() / connect()
    pipe = Pipeline()
    pipe.add_component("router",               QueryRouter())
    pipe.add_component("text_embedder",        SentenceTransformersTextEmbedder(
                                                   model="sentence-transformers/all-MiniLM-L6-v2"))
    pipe.add_component("retriever",            FAISSEmbeddingRetriever(
                                                   document_store=store, top_k=5))
    pipe.add_component("refusal",              OutOfScopeResponder())
    pipe.add_component("doc_grader",           DocumentGrader())
    pipe.add_component("answer_builder",       AnswerBuilder())
    pipe.add_component("hallucination_grader", HallucinationGrader(max_retries=2))
    pipe.add_component("answer_grader",        AnswerGrader())
    pipe.add_component("reformulator",         QueryReformulator())

    # Router outputs
    pipe.connect("router.course_query",            "text_embedder.text")
    pipe.connect("router.out_of_scope_query",      "refusal.query")

    # Retrieval chain
    pipe.connect("text_embedder.embedding",        "retriever.query_embedding")
    pipe.connect("retriever.documents",            "doc_grader.documents")
    pipe.connect("router.course_query",            "doc_grader.query")

    # Graded docs → answer generation
    pipe.connect("doc_grader.relevant_documents",  "answer_builder.documents")
    pipe.connect("doc_grader.query",               "answer_builder.query")

    # Answer → hallucination check
    pipe.connect("answer_builder.query",           "hallucination_grader.query")
    pipe.connect("answer_builder.answer",          "hallucination_grader.answer")
    pipe.connect("answer_builder.documents",       "hallucination_grader.documents")
    pipe.connect("answer_builder.sources",         "hallucination_grader.sources")

    # Hallucination check → answer grading
    pipe.connect("hallucination_grader.query",     "answer_grader.query")
    pipe.connect("hallucination_grader.answer",    "answer_grader.answer")
    pipe.connect("hallucination_grader.documents", "answer_grader.documents")
    pipe.connect("hallucination_grader.sources",   "answer_grader.sources")

    # Reformulation (for irrelevant docs or bad answers)
    pipe.connect("doc_grader.irrelevant_query",    "reformulator.query")
    pipe.connect("answer_grader.reformulate_query","reformulator.query")

    return pipe
```

**Usage:**

```python
pipe = build_rag_pipeline(
    faiss_index_path="data/faiss_store/faiss_index.faiss",
    db_url="sqlite:///data/faiss_store/faiss_document_store.db",
)

result = pipe.run({"router": {"query": "What is backpropagation?"}})
# Output is keyed by component name
out = result.get("answer_grader", result.get("refusal", {}))
print(out.get("answer"))
print(out.get("sources"))
```

### Step 14: LLM server for consumer device

Run `llama-cpp-python` as a local OpenAI-compatible server:

```bash
python -m llama_cpp.server \
    --model ./models/deepseek-8b-q4_K_M.gguf \
    --n_ctx 4096 \
    --n_threads 8 \
    --host 127.0.0.1 \
    --port 8000
```

All custom Haystack components call `http://localhost:8000/v1/chat/completions`.

**Memory budget (16 GB laptop):**

| Component | RAM |
|---|---|
| OS + system | ~3 GB |
| LLM (Q4_K_M, 4 K ctx) | ~6 GB |
| FAISSDocumentStore (HNSW, 50 K chunks) | ~0.15 GB |
| `sentence-transformers` (MiniLM) | ~0.1 GB |
| Python + Haystack pipeline overhead | ~0.5 GB |
| **Total** | **~9.75 GB** |
| **Headroom** | **~6.25 GB** ✅ |

### Step 15: Implement caching mechanisms

1. **Embedding cache** (Step 7) — skip re-embedding unchanged documents via content hash lookup.
2. **KV-cache reuse** — `llama-cpp-python` manages KV-cache within a session. Configure `n_ctx=4096`, `n_batch=512`.
3. **Query result cache** — wrap the pipeline's `run()` method:

```python
from cachetools import TTLCache
import hashlib

_query_cache = TTLCache(maxsize=100, ttl=3600)

def cached_query(pipe, query: str):
    key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    if key in _query_cache:
        return _query_cache[key]
    result = pipe.run({"router": {"query": query}})
    _query_cache[key] = result
    return result
```

---

## Phase 5 — UI & Demo (Week 4)

### Step 16: Build Gradio interface

**File: `src/app.py`**

**Tabs:**

1. **Chat** — question input, streaming answer, source citations (PDF name + page).
2. **Documents** — upload/remove PDFs → triggers the indexing pipeline (`DocumentCleaner` → `DocumentSplitter` → `SentenceTransformersDocumentEmbedder` → `DocumentWriter`) → shows ingestion status.
3. **Monitor** — real-time display of component execution trace (Router decision, retrieval scores, grading decisions, retry count). Haystack 2.x `Pipeline.run()` returns per-component output dicts for inspection.
4. **Settings** — adjustable parameters: Top-K, chunking strategy (`split_by`, `split_length`), temperature, max retries.

```python
import gradio as gr
from src.pipeline import build_rag_pipeline

pipe = build_rag_pipeline(
    faiss_index_path="data/faiss_store/faiss_index.faiss",
    db_url="sqlite:///data/faiss_store/faiss_document_store.db",
)

def answer_question(query: str, top_k: int = 5):
    result = pipe.run({
        "router": {"query": query},
        "retriever": {"top_k": top_k},
    })
    out = result.get("answer_grader", result.get("refusal", {}))
    answer = out.get("answer", "No answer generated.")
    sources = out.get("sources", [])
    source_text = "\n".join(
        f"- {s['source']}, page {s['page']}" for s in sources
    )
    return answer, source_text

with gr.Blocks(title="Course QA — Offline RAG") as demo:
    gr.Markdown("# 📚 Course QA Assistant (Fully Offline)")
    with gr.Tab("Chat"):
        query_input = gr.Textbox(label="Your Question", lines=2)
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K Documents")
        answer_output = gr.Textbox(label="Answer", lines=8)
        sources_output = gr.Textbox(label="Sources", lines=4)
        submit_btn = gr.Button("Ask")
        submit_btn.click(answer_question, [query_input, top_k_slider],
                         [answer_output, sources_output])
    with gr.Tab("Documents"):
        gr.Markdown("Upload PDFs to add to the knowledge base.")
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        ingest_status = gr.Textbox(label="Status", interactive=False)
    with gr.Tab("Settings"):
        gr.Markdown("Adjust pipeline parameters.")

demo.launch(server_name="127.0.0.1", server_port=7860)
```

---

## Phase 6 — Evaluation & Benchmarking (Week 4–5)

### Step 17: Build evaluation dataset

- Manually create **50–100 QA pairs** from course materials.
- Each entry: `{question, ground_truth_answer, source_doc, source_page, category}`.
- Categories: factual recall, conceptual understanding, multi-document reasoning, out-of-scope.
- Store as `eval/eval_dataset.json`.

### Step 18: Retrieval evaluation

Metrics (computed per query, averaged):

| Metric | Formula | What it measures |
|---|---|---|
| **Precision@K** | (relevant in top-K) / K | Retrieval accuracy |
| **Recall@K** | (relevant in top-K) / (total relevant) | Retrieval coverage |
| **MRR** | mean(1 / rank_of_first_relevant) | Ranking quality |

Run across all combinations:
- 3 chunking strategies × 3 chunk sizes × 3 K values × 4 index types = **108 configurations**
- Automate with `eval/run_retrieval_eval.py` → outputs CSV.

**Haystack 2.x native evaluation** — use built-in evaluator components from `haystack.components.evaluators`:

```python
from haystack.components.evaluators import DocumentMRREvaluator, DocumentRecallEvaluator

mrr_eval = DocumentMRREvaluator()
recall_eval = DocumentRecallEvaluator(mode="single_hit")

# Run retrieval pipeline on eval set, then compute metrics
mrr_result = mrr_eval.run(
    ground_truth_documents=ground_truth_docs,
    retrieved_documents=retrieved_docs,
)
```

### Step 19: Generation evaluation

| Metric | Method |
|---|---|
| **Answer accuracy** | Exact match + fuzzy match (token F1) against ground truth |
| **Relevance** | LLM-as-judge: "Does this answer address the question?" (binary) |
| **Factual consistency** | LLM-as-judge: "Is this answer supported by the provided context?" (binary) |

Use the DGX Spark's FP16 model (highest quality) to judge the Q4_K_M model's outputs.

### Step 20: System performance benchmarks

**File: `eval/run_system_bench.py`**

| Metric | How to measure |
|---|---|
| Time-to-first-token (TTFT) | Timestamp from query submission to first streamed token |
| Total generation time | Full response completion time |
| End-to-end query time | Including retrieval, grading, generation, all retries |
| Peak memory (RSS) | `psutil.Process().memory_info().rss` at peak |
| Tokens/second | Output tokens / generation time |

Run across quant levels on **both** DGX Spark and consumer laptop:

| Config | Device | Expected tok/s | RAM |
|---|---|---|---|
| F16 | DGX Spark | ~40–60 | ~17 GB |
| Q8_0 | DGX Spark | ~60–90 | ~9.5 GB |
| Q4_K_M | DGX Spark | ~80–120 | ~6 GB |
| Q4_K_M | Laptop (CPU) | ~8–15 | ~6 GB |
| Q4_0 | Laptop (CPU) | ~10–18 | ~5.5 GB |

---

## Phase 7 — Optimization & Report (Week 5–6)

### Step 21: Document three system-level optimizations

**Optimization 1: Model Quantization (FP16 → Q4_K_M)**
- Memory reduction: ~16 GB → ~4.9 GB file / ~6 GB runtime (**~2.8× savings**)
- Speed improvement: measure tok/s increase from FP16 to Q4_K_M on same hardware
- Quality impact: measure accuracy delta on eval set
- Makes deployment on 16 GB laptop **possible** (FP16 would not fit)

**Optimization 2: FAISS Index Acceleration (Flat → HNSW via Haystack)**
- Measure search latency: `FAISSDocumentStore(faiss_index_factory_str="Flat")` vs `"HNSW"` at N = 10 K, 25 K, 50 K chunks
- Expected: HNSW is 10–50× faster at 50 K with >95% recall
- Switching index type is a **one-line change** in Haystack — demonstrates framework benefit

**Optimization 3: Embedding Cache**
- Measure time to ingest 50+ PDFs cold (no cache) vs warm (cache hit on unchanged docs)
- Expected: near-zero time for unchanged documents
- Enables incremental updates when new lecture slides are added

### Step 22: Final deliverables

1. **Working offline demo** — Gradio app answering course questions in real-time on a laptop.
2. **Benchmark report** with tables and charts:
   - Chunking strategy comparison (retrieval metrics)
   - Quant level comparison (speed, memory, quality)
   - Index type comparison (search latency, recall)
   - Optimization gains (before/after measurements)
3. **Source code** — clean, documented, with README for setup and reproduction.
4. **Demo script** — 5-minute walkthrough: cold start → upload PDF → ask questions → show monitoring → offline verification.

---

## Project Structure

```
project/
├── src/
│   ├── ingest.py           # PDF extraction → Haystack Documents
│   ├── store.py            # FAISSDocumentStore + embedder/retriever factory
│   ├── indexer.py          # DocumentSplitter indexing pipeline
│   ├── nodes.py            # Custom @component classes (Router, Graders, Builder)
│   ├── pipeline.py         # Haystack Pipeline wiring
│   ├── cache.py            # Embedding cache + query cache
│   └── app.py              # Gradio UI
├── eval/
│   ├── eval_dataset.json   # QA evaluation set
│   ├── run_retrieval_eval.py
│   ├── run_generation_eval.py
│   └── run_system_bench.py
├── scripts/
│   └── quantization/       # Shell scripts for model conversion and quantization (run on DGX Spark)
│       ├── convert_to_gguf.sh   # Download model from HF + convert HF → GGUF FP16
│       ├── quantize_all.sh      # Loop over quant levels (q8_0, q5_K_M, q4_K_M, q4_0)
│       └── validate_quants.sh   # Spot-check each GGUF with a test prompt
├── data/
│   ├── raw/                # Original PDFs
│   ├── processed/          # Extracted text JSON
│   ├── faiss_store/        # FAISSDocumentStore persistence
│   └── embeddings_cache/   # Embedding cache
├── models/                 # GGUF files
├── deploy/                 # Packaged deployment artifacts
├── requirements.txt
└── README.md
```

### `requirements.txt`

```
haystack-ai
faiss-haystack
sentence-transformers
llama-cpp-python
pymupdf
gradio
pydantic
tqdm
psutil
cachetools
requests
```

---

## Why Haystack over raw LangChain/FAISS

| Concern | Haystack Advantage |
|---|---|
| **Document store** | `FAISSDocumentStore` bundles FAISS index + SQLite metadata in one object — no manual sidecar JSON |
| **Preprocessing** | Built-in `DocumentCleaner` + `DocumentSplitter` components with `split_by`, `split_length`, `split_overlap` |
| **Embedding** | `SentenceTransformersDocumentEmbedder` / `SentenceTransformersTextEmbedder` wrap `sentence-transformers` directly |
| **Index switching** | Change `faiss_index_factory_str` from `"Flat"` to `"HNSW"` — one parameter, no code change |
| **Pipeline** | `Pipeline` class with `add_component()` / `connect()` manages component graph and data flow |
| **Evaluation** | `DocumentMRREvaluator` / `DocumentRecallEvaluator` components plug directly into evaluation workflows |
| **Persistence** | `store.save()` / `FAISSDocumentStore.load()` — portable, single-command serialization |

---

## Verification Checklist

- [ ] `haystack-ai`, `faiss-haystack`, and `sentence-transformers` install cleanly on both machines
- [ ] PDF ingestion produces valid Haystack `Document` objects with metadata
- [ ] `DocumentSplitter` splits produce chunks within expected token ranges
- [ ] Indexing pipeline (`DocumentCleaner` → `DocumentSplitter` → `SentenceTransformersDocumentEmbedder` → `DocumentWriter`) completes without error
- [ ] `FAISSDocumentStore.save()` / `.load()` round-trips correctly
- [ ] All GGUF quant levels produce coherent output
- [ ] Custom `@component` classes (Router, DocumentGrader, HallucinationGrader, AnswerGrader) return correct output dicts
- [ ] Full pipeline completes: Route → Retrieve → Grade → Generate → Hallucination Check → Answer Check
- [ ] Gradio demo runs fully offline (disconnect network and verify)
- [ ] Peak memory stays under **14 GB** on 16 GB laptop
- [ ] Benchmark script produces complete CSV with all metrics
- [ ] Precision@5 ≥ 0.6 on eval set (sanity threshold)
- [ ] End-to-end query time < 60 s on laptop with Q4_K_M
