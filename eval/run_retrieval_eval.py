"""
eval/run_retrieval_eval.py — Retrieval benchmark for DocumentChunker
====================================================================

Benchmarks Precision@K, Recall@K, and MRR across:
  - chunking strategies  (fixed, sentence)
  - chunk sizes          (256, 512, 1024 characters)
  - overlap values       (0, 10 %, 20 % of chunk_size)
  - top-K values         (1, 3, 5, 10)

Expects eval_dataset.json in the same directory:

    [
      {
        "question":       "What is gradient descent?",
        "answer":         "...",
        "source_doc":     "textbook/chapter3.pdf",
        "source_page":    12,
        "category":       "optimisation"
      },
      ...
    ]

Usage
-----
    # Quick smoke-test (small sweep)
    python eval/run_retrieval_eval.py --quick

    # Full sweep (takes longer)
    python eval/run_retrieval_eval.py

    # Custom data / model
    python eval/run_retrieval_eval.py \\
        --data-dir data/raw \\
        --eval-dataset eval/eval_dataset.json \\
        --embed-model sentence-transformers/all-MiniLM-L6-v2 \\
        --output eval/retrieval_results.json

Results are written to eval/retrieval_results.json and a summary table is
printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — fail gracefully if haystack/sentence-transformers not found
# ---------------------------------------------------------------------------
try:
    from haystack import Document, Pipeline
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError as exc:
    raise SystemExit(
        "haystack-ai is required.  Install with:  pip install haystack-ai"
    ) from exc

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required.  Install with:  pip install pymupdf"
    ) from exc

# chunker lives one level up in src/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from chunker import ChunkerConfig, DocumentChunker, infer_doc_type  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question:    str
    answer:      str
    source_doc:  str          # relative path within data/raw
    source_page: Optional[int] = None
    category:    Optional[str] = None


@dataclass
class RetrievalMetrics:
    strategy:   str
    chunk_size: int
    overlap:    int
    top_k:      int
    precision:  float
    recall:     float
    mrr:        float
    latency_ms: float         # average query latency in milliseconds


# ---------------------------------------------------------------------------
# PDF ingestion (minimal — no dependency on src/ingest.py yet)
# ---------------------------------------------------------------------------

def load_pdfs_from_dir(data_dir: Path) -> List[Document]:
    """Walk *data_dir* recursively, extract text from each PDF."""
    docs: List[Document] = []
    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.warning("Could not open %s: %s", pdf_path, exc)
            continue

        rel_path = str(pdf_path.relative_to(data_dir))
        doc_type = infer_doc_type(str(pdf_path))

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    content=text,
                    meta={
                        "file_path":   rel_path,
                        "source":      rel_path,
                        "page_number": page_num,
                        "doc_type":    doc_type,
                    }
                ))
        pdf.close()

    logger.info("Loaded %d pages from %s", len(docs), data_dir)
    return docs


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    raw_docs:    List[Document],
    cfg:         ChunkerConfig,
    embed_model: str,
) -> Tuple[InMemoryDocumentStore, SentenceTransformersTextEmbedder]:
    """Chunk → embed → write to an in-memory store.  Returns (store, query_embedder)."""

    # 1. Chunk
    chunker  = DocumentChunker.from_config(cfg)
    result   = chunker.run(documents=raw_docs)
    chunks   = result["documents"]
    logger.info("  Chunked into %d chunks", len(chunks))

    # 2. Embed documents
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embed_model)
    doc_embedder.warm_up()
    embedded = doc_embedder.run(documents=chunks)["documents"]

    # 3. Store
    store = InMemoryDocumentStore()
    store.write_documents(embedded)

    # 4. Query embedder (same model, reused across queries)
    query_embedder = SentenceTransformersTextEmbedder(model=embed_model)
    query_embedder.warm_up()

    return store, query_embedder


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def _is_hit(retrieved: List[Document], sample: EvalSample) -> bool:
    """
    A retrieved chunk is a hit if its source path contains the sample's
    source_doc string (partial match; robust to absolute-vs-relative paths).
    """
    for doc in retrieved:
        source = doc.meta.get("source", "") or doc.meta.get("file_path", "")
        if sample.source_doc in source or source in sample.source_doc:
            return True
    return False


def _reciprocal_rank(retrieved: List[Document], sample: EvalSample) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        source = doc.meta.get("source", "") or doc.meta.get("file_path", "")
        if sample.source_doc in source or source in sample.source_doc:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    samples:        List[EvalSample],
    store:          InMemoryDocumentStore,
    query_embedder: SentenceTransformersTextEmbedder,
    cfg:            ChunkerConfig,
    top_k:          int,
) -> RetrievalMetrics:
    retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=top_k)

    hits      = 0
    total_rr  = 0.0
    latencies: List[float] = []

    for sample in samples:
        t0  = time.perf_counter()
        emb = query_embedder.run(text=sample.question)
        res = retriever.run(query_embedding=emb["embedding"])
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = res["documents"]
        if _is_hit(retrieved, sample):
            hits += 1
        total_rr += _reciprocal_rank(retrieved, sample)

    n         = len(samples)
    precision = hits / n
    recall    = precision           # binary relevance → P@K == R@K here
    mrr       = total_rr / n
    avg_lat   = sum(latencies) / len(latencies) if latencies else 0.0

    return RetrievalMetrics(
        strategy=cfg.strategy,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        top_k=top_k,
        precision=precision,
        recall=recall,
        mrr=mrr,
        latency_ms=avg_lat,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval benchmark for DocumentChunker")
    p.add_argument("--data-dir",      default="data/raw",
                   help="Root directory of raw PDFs (default: data/raw)")
    p.add_argument("--eval-dataset",  default="eval/eval_dataset.json",
                   help="Path to eval_dataset.json")
    p.add_argument("--embed-model",   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="SentenceTransformers model name")
    p.add_argument("--output",        default="eval/retrieval_results.json",
                   help="Where to write JSON results")
    p.add_argument("--quick",         action="store_true",
                   help="Run a small subset of the sweep for smoke-testing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ data
    data_dir  = Path(args.data_dir)
    eval_path = Path(args.eval_dataset)

    if not data_dir.exists():
        raise SystemExit(f"data-dir not found: {data_dir}")
    if not eval_path.exists():
        raise SystemExit(f"eval-dataset not found: {eval_path}")

    raw_docs = load_pdfs_from_dir(data_dir)
    if not raw_docs:
        raise SystemExit(f"No PDFs found under {data_dir}.")

    with eval_path.open() as f:
        samples = [EvalSample(**s) for s in json.load(f)]
    logger.info("Loaded %d eval samples", len(samples))

    # --------------------------------------------------------------- sweep grid
    if args.quick:
        configs   = [ChunkerConfig("fixed", 512, 64)]
        k_values  = [5]
    else:
        configs = [
            ChunkerConfig(strategy, size, int(size * overlap_frac))
            for strategy     in ("fixed", "sentence")
            for size         in (256, 512, 1024)
            for overlap_frac in (0.0, 0.1, 0.2)
        ]
        k_values = [1, 3, 5, 10]

    # ------------------------------------------------------------------ sweep
    all_results: List[RetrievalMetrics] = []

    for cfg in configs:
        label = f"strategy={cfg.strategy} | size={cfg.chunk_size} | overlap={cfg.overlap}"
        logger.info("Building index: %s", label)
        store, query_embedder = build_index(raw_docs, cfg, args.embed_model)

        for k in k_values:
            logger.info("  Evaluating top_k=%d …", k)
            metrics = evaluate_retrieval(samples, store, query_embedder, cfg, top_k=k)
            all_results.append(metrics)
            logger.info(
                "  P@%d=%.3f  R@%d=%.3f  MRR=%.3f  lat=%.1fms",
                k, metrics.precision,
                k, metrics.recall,
                metrics.mrr,
                metrics.latency_ms,
            )

    # ----------------------------------------------------------------- output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    logger.info("Results written to %s", out_path)

    # ----------------------------------------------------------- summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<10} {'Size':>6} {'Overlap':>8} {'K':>4} "
          f"{'P@K':>7} {'R@K':>7} {'MRR':>7} {'Lat(ms)':>9}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: -x.mrr):
        print(
            f"{r.strategy:<10} {r.chunk_size:>6} {r.overlap:>8} {r.top_k:>4} "
            f"{r.precision:>7.3f} {r.recall:>7.3f} {r.mrr:>7.3f} {r.latency_ms:>9.1f}"
        )
    print("=" * 80)

    best = max(all_results, key=lambda x: x.mrr)
    print(
        f"\nBest config by MRR:  strategy={best.strategy}  "
        f"chunk_size={best.chunk_size}  overlap={best.overlap}  "
        f"top_k={best.top_k}  MRR={best.mrr:.3f}"
    )


if __name__ == "__main__":
    main()
