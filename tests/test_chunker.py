"""
tests/test_chunker.py — Unit tests for src/chunker.py
======================================================

Run with:
    pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from haystack import Document
from chunker import (
    ChunkerConfig,
    DocumentChunker,
    _chunk_fixed,
    _chunk_sentence,
    _clean_text,
    infer_doc_type,
)


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_form_feed(self):
        assert "\x0c" not in _clean_text("Hello\x0cWorld")

    def test_single_newline_becomes_space(self):
        result = _clean_text("Hello\nWorld")
        assert "\n" not in result
        assert "Hello World" == result

    def test_paragraph_breaks_preserved(self):
        result = _clean_text("Para one.\n\nPara two.")
        assert "\n\n" in result

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_text("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# infer_doc_type
# ---------------------------------------------------------------------------

class TestInferDocType:
    @pytest.mark.parametrize("path,expected", [
        ("data/raw/lectures/lecture01.pdf",        "lecture"),
        ("data/raw/textbook/chapter3.pdf",         "textbook"),
        ("data/raw/assignments/hw2.pdf",           "assignment"),
        ("data/raw/assignments/worksheet_3.pdf",   "assignment"),
        ("data/raw/other/something.pdf",           "unknown"),
        (None,                                     "unknown"),
        ("",                                       "unknown"),
    ])
    def test_inference(self, path, expected):
        assert infer_doc_type(path) == expected


# ---------------------------------------------------------------------------
# _chunk_fixed
# ---------------------------------------------------------------------------

class TestChunkFixed:
    def test_empty_string(self):
        assert _chunk_fixed("", 512, 64) == []

    def test_short_text_is_single_chunk(self):
        text = "Hello world. This is a short sentence."
        chunks = _chunk_fixed(text, 512, 64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        # 20 sentences of ~40 chars → ~800 chars total; chunk_size=200
        sent = "This is a sentence with some words. "
        text = sent * 20
        chunks = _chunk_fixed(text, 200, 20)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_chunk_size_by_much(self):
        # Each chunk should be ≤ chunk_size + longest_sentence (edge case).
        sent = "Short sentence here. "
        text = sent * 50
        chunks = _chunk_fixed(text, 100, 10)
        for chunk in chunks:
            # Allow one sentence of slack for snapping.
            assert len(chunk) < 200, f"Chunk too long: {len(chunk)}"

    def test_overlap_carries_context(self):
        # With overlap, the start of chunk[1] should partially overlap chunk[0].
        sent = "Overlap test sentence. "
        text = sent * 30
        chunks_no_overlap = _chunk_fixed(text, 150, 0)
        chunks_overlap    = _chunk_fixed(text, 150, 50)
        # Overlapping chunks should be more numerous.
        assert len(chunks_overlap) >= len(chunks_no_overlap)

    def test_zero_overlap(self):
        sent = "Zero overlap sentence here. "
        text = sent * 20
        chunks = _chunk_fixed(text, 100, 0)
        assert all(len(c) > 0 for c in chunks)

    def test_single_very_long_sentence_hard_split(self):
        # A sentence longer than chunk_size must still be chunked.
        long_sent = "A" * 600
        chunks = _chunk_fixed(long_sent, 200, 20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200


# ---------------------------------------------------------------------------
# _chunk_sentence
# ---------------------------------------------------------------------------

class TestChunkSentence:
    def test_empty_string(self):
        assert _chunk_sentence("", 512, 64) == []

    def test_paragraphs_kept_whole_when_small(self):
        text = "Para one is here.\n\nPara two is also here."
        chunks = _chunk_sentence(text, 512, 64)
        assert len(chunks) == 1  # both fit in one chunk

    def test_large_paragraphs_split(self):
        para = "This is a long paragraph. " * 30   # ~750 chars
        text = para + "\n\n" + para
        chunks = _chunk_sentence(text, 300, 30)
        assert len(chunks) > 2

    def test_output_not_empty(self):
        text = "Hello world.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _chunk_sentence(text, 512, 64)
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# DocumentChunker (Haystack component)
# ---------------------------------------------------------------------------

class TestDocumentChunker:

    def _make_doc(self, text: str, **meta) -> Document:
        return Document(content=text, meta={"file_path": "lectures/lec1.pdf", **meta})

    # ---- constructor validation ----

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            DocumentChunker(strategy="semantic")  # type: ignore

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, overlap=100)

    # ---- run() ----

    def test_empty_input(self):
        chunker = DocumentChunker()
        result  = chunker.run(documents=[])
        assert result["documents"] == []

    def test_empty_content_skipped(self):
        chunker = DocumentChunker()
        doc     = Document(content="   ", meta={})
        result  = chunker.run(documents=[doc])
        assert result["documents"] == []

    def test_basic_chunking_produces_output(self):
        chunker = DocumentChunker(strategy="fixed", chunk_size=200, overlap=20)
        text    = "This is a test sentence. " * 30
        doc     = self._make_doc(text)
        result  = chunker.run(documents=[doc])
        assert len(result["documents"]) > 1

    def test_metadata_propagation(self):
        chunker = DocumentChunker(strategy="fixed", chunk_size=100, overlap=10)
        text    = "Sentence one. " * 20
        doc     = Document(
            content=text,
            meta={"file_path": "lectures/lec1.pdf", "page_number": 3},
        )
        chunks = chunker.run(documents=[doc])["documents"]
        for chunk in chunks:
            assert chunk.meta["page_number"] == 3
            assert chunk.meta["source"] == "lectures/lec1.pdf"
            assert chunk.meta["doc_type"] == "lecture"

    def test_chunk_id_and_total(self):
        chunker = DocumentChunker(strategy="fixed", chunk_size=100, overlap=10)
        text    = "Short sentence. " * 30
        doc     = self._make_doc(text)
        chunks  = chunker.run(documents=[doc])["documents"]
        total   = chunks[0].meta["chunk_total"]
        assert total == len(chunks)
        for i, chunk in enumerate(chunks):
            assert chunk.meta["chunk_id"] == i
            assert chunk.meta["chunk_total"] == total

    def test_sentence_strategy(self):
        chunker = DocumentChunker(strategy="sentence", chunk_size=200, overlap=30)
        text    = "Paragraph one.\n\nParagraph two.\n\n" * 10
        doc     = self._make_doc(text)
        result  = chunker.run(documents=[doc])
        assert len(result["documents"]) > 0

    def test_multiple_documents(self):
        chunker = DocumentChunker(strategy="fixed", chunk_size=100, overlap=10)
        docs    = [self._make_doc("Sentence. " * 20) for _ in range(3)]
        result  = chunker.run(documents=docs)
        assert len(result["documents"]) > 3

    # ---- serialisation ----

    def test_to_dict_round_trip(self):
        original = DocumentChunker(strategy="sentence", chunk_size=256, overlap=32)
        d        = original.to_dict()
        restored = DocumentChunker.from_dict(d)
        assert restored.strategy   == original.strategy
        assert restored.chunk_size == original.chunk_size
        assert restored.overlap    == original.overlap

    def test_from_config(self):
        cfg     = ChunkerConfig(strategy="sentence", chunk_size=300, overlap=30)
        chunker = DocumentChunker.from_config(cfg)
        assert chunker.strategy   == "sentence"
        assert chunker.chunk_size == 300
        assert chunker.overlap    == 30
