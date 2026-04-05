"""
src/chunker.py — Chunking strategies for CourseQA-RAG
======================================================

Provides a Haystack 2.x ``@component`` (DocumentChunker) that sits in the
indexing pipeline between ingest (raw Documents) and the embedder/writer.

Two strategies
--------------
fixed
    Fixed-size character window with sentence-boundary snapping and
    configurable overlap.  Best default for mixed content (slides + textbooks
    + assignments).  A chunk never cuts mid-sentence.

sentence
    Paragraph-first grouping that respects natural text boundaries, falling
    back to sentence grouping when a paragraph is too long.  Better for
    flowing textbook prose where paragraph structure is meaningful.

Wiring into indexer.py
-----------------------
    from chunker import DocumentChunker

    chunker = DocumentChunker(strategy="fixed", chunk_size=512, overlap=64)
    pipeline.add_component("chunker", chunker)
    pipeline.connect("cleaner.documents", "chunker.documents")
    pipeline.connect("chunker.documents",  "embedder.documents")

Defaults are tuned for all-MiniLM-L6-v2 (256-token limit ≈ 1 024 chars;
512 chars keeps us safely below that while leaving room for the prompt).

Priority order when choosing values:
    CLI / app config  >  ChunkerConfig dataclass  >  constructor defaults
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass (used by eval scripts and app.py)
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    """
    Serialisable configuration for DocumentChunker.

    Use this in your eval scripts to sweep parameters without rewriting
    the pipeline:

        for cfg in [
            ChunkerConfig("fixed",    chunk_size=256, overlap=32),
            ChunkerConfig("fixed",    chunk_size=512, overlap=64),
            ChunkerConfig("sentence", chunk_size=512, overlap=64),
        ]:
            chunker = DocumentChunker.from_config(cfg)
            ...
    """
    strategy:   Literal["fixed", "sentence"] = "fixed"
    chunk_size: int = 512   # characters  (512 chars ≈ 128 tokens for English)
    overlap:    int = 64    # characters  (~12 % of default chunk_size)


# ---------------------------------------------------------------------------
# Internal text-splitting utilities (zero extra dependencies)
# ---------------------------------------------------------------------------

# End-of-sentence: punctuation followed by whitespace + uppercase / accented.
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F\u0400-\u04FF])')

# Paragraph boundary: one or more blank lines.
_PARA_RE = re.compile(r'\n\s*\n')

# PDF artefacts common in lecture slides and scanned textbooks.
_CLEANUP_RE = re.compile(
    r'(\x0c'                    # form-feed (page break marker from PyMuPDF)
    r'|\u00ad'                  # soft hyphen
    r'|(?<!\n)\n(?!\n)'         # single newline inside a paragraph → space
    r')',
    re.UNICODE,
)


def _clean_text(text: str) -> str:
    """Light normalisation of PDF-extracted text before chunking."""
    text = _CLEANUP_RE.sub(lambda m: ' ' if m.group() == '\n' else '', text)
    # Collapse runs of whitespace (but preserve paragraph breaks → \n\n).
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    parts = _PARA_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    # Fall back to sentence splitting if no paragraph boundaries found.
    return parts if len(parts) > 1 else _split_sentences(text)


def _overlap_tail(parts: List[str], overlap: int) -> List[str]:
    """Return the minimal *suffix* of parts whose total length >= overlap."""
    if overlap <= 0 or not parts:
        return []
    tail: List[str] = []
    total = 0
    for part in reversed(parts):
        tail.insert(0, part)
        total += len(part) + 1
        if total >= overlap:
            break
    return tail


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Slide a window of *chunk_size* characters over *text*, snapping to the
    nearest sentence boundary.  *overlap* characters of context are carried
    forward from the previous chunk.

    Edge case: a single sentence longer than chunk_size is hard-split by
    characters (unavoidable for dense math / code blocks in textbooks).
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        # No sentence boundaries found — hard-split.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        s_len = len(sent) + 1  # +1 for the space that joins them

        # Single sentence exceeds budget — hard-split it.
        if s_len > chunk_size:
            if current:
                chunks.append(' '.join(current))
                current = _overlap_tail(current, overlap)
                current_len = sum(len(s) + 1 for s in current)
            for start in range(0, len(sent), chunk_size - max(overlap, 0)):
                piece = sent[start:start + chunk_size]
                if piece:
                    chunks.append(piece)
            # Seed overlap from the last hard-split piece.
            if chunks:
                seed = chunks[-1][-overlap:] if overlap else ''
                current = [seed] if seed else []
                current_len = len(seed) + 1
            continue

        # Would this sentence overflow the current chunk?
        if current_len + s_len > chunk_size and current:
            chunks.append(' '.join(current))
            current = _overlap_tail(current, overlap)
            current_len = sum(len(s) + 1 for s in current)

        current.append(sent)
        current_len += s_len

    if current:
        chunks.append(' '.join(current))

    return [c for c in chunks if c.strip()]


def _chunk_sentence(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Paragraph-first strategy.  Whole paragraphs are accumulated until the
    chunk budget is full; oversized paragraphs are recursively split with
    _chunk_fixed at sentence level.
    """
    if not text:
        return []

    paragraphs = _split_paragraphs(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        p_len = len(para) + 2  # +2 for '\n\n' joiner

        if p_len > chunk_size:
            # Flush accumulated buffer.
            if current:
                chunks.append('\n\n'.join(current))
                current = _overlap_tail(current, overlap)
                current_len = sum(len(p) + 2 for p in current)

            # Recursively split the oversized paragraph.
            sub = _chunk_fixed(para, chunk_size, overlap)
            chunks.extend(sub)

            # Seed overlap from last sub-chunk.
            if sub and overlap:
                seed = sub[-1][-overlap:]
                current = [seed] if seed else []
                current_len = len(seed) + 2
            continue

        if current_len + p_len > chunk_size and current:
            chunks.append('\n\n'.join(current))
            current = _overlap_tail(current, overlap)
            current_len = sum(len(p) + 2 for p in current)

        current.append(para)
        current_len += p_len

    if current:
        chunks.append('\n\n'.join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Doc-type inference from file path
# ---------------------------------------------------------------------------

_DOC_TYPE_MAP: Dict[str, re.Pattern] = {
    "lecture":    re.compile(r'lecture', re.IGNORECASE),
    "textbook":   re.compile(r'textbook|chapter', re.IGNORECASE),
    "assignment": re.compile(r'assignment|worksheet|hw|homework', re.IGNORECASE),
}


def infer_doc_type(source: Optional[str]) -> str:
    """Infer document category from a file path string."""
    if not source:
        return "unknown"
    for doc_type, pattern in _DOC_TYPE_MAP.items():
        if pattern.search(source):
            return doc_type
    return "unknown"


# ---------------------------------------------------------------------------
# Haystack 2.x component
# ---------------------------------------------------------------------------

@component
class DocumentChunker:
    """
    Splits Haystack Documents into smaller chunks for RAG indexing.

    Parameters
    ----------
    strategy : {"fixed", "sentence"}
        Chunking algorithm to use (see module docstring).
    chunk_size : int
        Target chunk size in **characters**.
        Default 512 ≈ 128 tokens; safe for all-MiniLM-L6-v2's 256-token cap.
    overlap : int
        Characters of carry-over from the previous chunk.
        64 (≈ 12 % of 512) is a good default.

    Output metadata per chunk
    -------------------------
    All parent metadata is propagated, plus:
        source      — file path (str)
        page_number — page where chunk originates (int, if set by ingest.py)
        doc_type    — "lecture" | "textbook" | "assignment" | "unknown"
        chunk_id    — zero-based index within the parent document (int)
        chunk_total — total number of chunks from this parent (int)
    """

    def __init__(
        self,
        strategy:   Literal["fixed", "sentence"] = "fixed",
        chunk_size: int = 512,
        overlap:    int = 64,
    ) -> None:
        if strategy not in ("fixed", "sentence"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'fixed' or 'sentence'.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if not (0 <= overlap < chunk_size):
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

        self.strategy   = strategy
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self._fn        = _chunk_fixed if strategy == "fixed" else _chunk_sentence

    # ------------------------------------------------------------------
    # Haystack serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return default_to_dict(
            self,
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunker":
        return default_from_dict(cls, data)

    @classmethod
    def from_config(cls, cfg: ChunkerConfig) -> "DocumentChunker":
        """Construct from a ChunkerConfig (convenience for eval sweeps)."""
        return cls(
            strategy=cfg.strategy,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
        )

    # ------------------------------------------------------------------
    # Component I/O
    # ------------------------------------------------------------------

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> dict:
        """
        Split each input Document into chunk Documents.

        Parameters
        ----------
        documents : List[Document]
            Raw documents from the cleaner / ingest step.

        Returns
        -------
        dict with key "documents" containing the list of chunk Documents.
        """
        if not documents:
            logger.warning("DocumentChunker received an empty document list.")
            return {"documents": []}

        output: List[Document] = []

        for doc in documents:
            text = doc.content or ""
            text = _clean_text(text)

            if not text:
                logger.debug(
                    "Skipping empty document: %s",
                    doc.meta.get("file_path", "<unknown>"),
                )
                continue

            raw_chunks = self._fn(text, self.chunk_size, self.overlap)

            source   = doc.meta.get("file_path") or doc.meta.get("source", "")
            doc_type = doc.meta.get("doc_type") or infer_doc_type(source)
            total    = len(raw_chunks)

            for idx, chunk_text in enumerate(raw_chunks):
                meta = {
                    **doc.meta,          # propagate parent metadata
                    "source":      source,
                    "doc_type":    doc_type,
                    "chunk_id":    idx,
                    "chunk_total": total,
                    # page_number is already in doc.meta if ingest.py set it.
                    # We do NOT override it here; it marks where the chunk starts.
                }
                output.append(Document(content=chunk_text, meta=meta))

        logger.info(
            "DocumentChunker [strategy=%s | chunk_size=%d | overlap=%d] "
            "%d docs → %d chunks",
            self.strategy,
            self.chunk_size,
            self.overlap,
            len(documents),
            len(output),
        )
        return {"documents": output}
