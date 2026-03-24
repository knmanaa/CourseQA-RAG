import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from pathlib import Path
import json

from haystack import Document
import numpy as np


class ChunkingMethod(Enum):
    """Enumeration of available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    # Basic parameters
    method: ChunkingMethod = ChunkingMethod.FIXED_SIZE
    chunk_size: int = 500  # characters or tokens
    chunk_overlap: int = 50
    split_by: str = "word"  # "word", "sentence", "paragraph", "page"
    
    # Semantic chunking
    similarity_threshold: float = 0.85
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    
    # Recursive chunking
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])
    
    # Sliding window
    window_stride: int = 250
    
    # Hierarchical
    parent_chunk_size: int = 2000
    child_chunk_size: int = 250
    
    # Metadata
    preserve_metadata: bool = True
    add_chunk_index: bool = True
    track_overlap: bool = True


class TextSplitter:
    """Base class for text splitters."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        raise NotImplementedError
    
    def split_document(self, document: Document) -> List[Document]:
        """Split a Haystack Document into chunks."""
        chunks = self.split_text(document.content)
        
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            metadata = document.meta.copy()
            if self.config.add_chunk_index:
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                metadata["chunk_strategy"] = self.config.method.value
            
            chunk_docs.append(Document(
                content=chunk,
                meta=metadata,
                embedding=document.embedding if i == 0 else None  # Only first chunk keeps embedding
            ))
        
        return chunk_docs


class FixedSizeSplitter(TextSplitter):
    """Fixed-size chunking by characters or words."""
    
    def split_text(self, text: str) -> List[str]:
        chunks = []
        split_by = self.config.split_by
        
        if split_by == "word":
            words = text.split()
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
                    
        elif split_by == "sentence":
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = self._merge_by_size(sentences)
            
        elif split_by == "paragraph":
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = self._merge_by_size(paragraphs)
            
        else:  # character
            for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
                chunk = text[i:i + self.config.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        
        return chunks
    
    def _merge_by_size(self, segments: List[str]) -> List[str]:
        """Merge segments to reach target chunk size."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for seg in segments:
            seg_size = len(seg)
            if current_size + seg_size <= self.config.chunk_size:
                current_chunk.append(seg)
                current_size += seg_size
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [seg]
                current_size = seg_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class SemanticSplitter(TextSplitter):
    """Semantic chunking based on content similarity."""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._cache = {}  # Cache for embeddings
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (simplified - would use actual embedding model)."""
        # In production, use a proper embedding model
        # This is a placeholder using TF-IDF-like approach
        words = text.split()
        if not words:
            return np.zeros(384)
        
        # Simple hash-based embedding for demonstration
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = np.frombuffer(hash_bytes[:384], dtype=np.uint8) / 255.0
        return embedding.astype(np.float32)
    
    def split_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            return [text]
        
        # Group sentences by semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = self._get_embedding(sentences[0])
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_embedding = self._get_embedding(sent)
            
            # Compute similarity
            similarity = np.dot(current_embedding, sent_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(sent_embedding) + 1e-8
            )
            
            # Check size constraints
            current_size = sum(len(s) for s in current_chunk)
            sent_size = len(sent)
            
            if (similarity >= self.config.similarity_threshold and 
                current_size + sent_size <= self.config.max_chunk_size):
                current_chunk.append(sent)
                # Update embedding as average
                current_embedding = (current_embedding + sent_embedding) / 2
            else:
                # Save current chunk if it meets minimum size
                if current_size >= self.config.min_chunk_size:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_embedding = sent_embedding
        
        # Add last chunk
        if current_chunk and sum(len(s) for s in current_chunk) >= self.config.min_chunk_size:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class RecursiveSplitter(TextSplitter):
    """Recursive splitting with multiple separators."""
    
    def split_text(self, text: str) -> List[str]:
        chunks = self._split_recursive(text, self.config.separators)
        return chunks
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Split into characters
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_size = len(split)
            
            if current_size + split_size <= self.config.chunk_size:
                current_chunk.append(split)
                current_size += split_size
            else:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if len(chunk_text) > self.config.chunk_size:
                        # Chunk is still too large, try deeper recursion
                        sub_chunks = self._split_recursive(chunk_text, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk_text)
                
                current_chunk = [split]
                current_size = split_size
        
        # Handle last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.config.chunk_size:
                sub_chunks = self._split_recursive(chunk_text, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks


class SlidingWindowSplitter(TextSplitter):
    """Sliding window chunking with stride."""
    
    def split_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.window_stride):
            chunk = " ".join(words[i:i + self.config.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks


class HierarchicalSplitter(TextSplitter):
    """Hierarchical chunking with parent-child relationships."""
    
    def split_text(self, text: str) -> List[str]:
        # First create parent chunks
        parent_splitter = FixedSizeSplitter(ChunkingConfig(
            method=ChunkingMethod.FIXED_SIZE,
            chunk_size=self.config.parent_chunk_size,
            split_by="paragraph"
        ))
        parent_chunks = parent_splitter.split_text(text)
        
        # Then split each parent into children
        child_splitter = FixedSizeSplitter(ChunkingConfig(
            method=ChunkingMethod.FIXED_SIZE,
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.chunk_overlap
        ))
        
        all_chunks = []
        for parent in parent_chunks:
            children = child_splitter.split_text(parent)
            all_chunks.extend(children)
        
        return all_chunks


class HybridSplitter(TextSplitter):
    """Hybrid approach combining multiple strategies."""
    
    def split_text(self, text: str) -> List[str]:
        # Try semantic first
        semantic = SemanticSplitter(self.config)
        chunks = semantic.split_text(text)
        
        # If any chunk is too large, split recursively
        recursive = RecursiveSplitter(self.config)
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) > self.config.max_chunk_size:
                sub_chunks = recursive.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks


class ChunkingStrategy:
    """Main chunking strategy manager."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._splitter = self._create_splitter()
        
    def _create_splitter(self) -> TextSplitter:
        """Create the appropriate splitter based on config."""
        splitters = {
            ChunkingMethod.FIXED_SIZE: FixedSizeSplitter,
            ChunkingMethod.SEMANTIC: SemanticSplitter,
            ChunkingMethod.RECURSIVE: RecursiveSplitter,
            ChunkingMethod.SLIDING_WINDOW: SlidingWindowSplitter,
            ChunkingMethod.HIERARCHICAL: HierarchicalSplitter,
            ChunkingMethod.HYBRID: HybridSplitter,
        }
        
        splitter_class = splitters.get(self.config.method)
        if not splitter_class:
            raise ValueError(f"Unknown chunking method: {self.config.method}")
        
        return splitter_class(self.config)
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document."""
        return self._splitter.split_document(document)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents."""
        chunked_docs = []
        for doc in documents:
            chunked_docs.extend(self.chunk_document(doc))
        return chunked_docs
    
    def get_chunk_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about chunked documents."""
        chunk_lengths = [len(doc.content) for doc in documents]
        
        return {
            "total_chunks": len(documents),
            "avg_chunk_size": np.mean(chunk_lengths),
            "std_chunk_size": np.std(chunk_lengths),
            "min_chunk_size": np.min(chunk_lengths),
            "max_chunk_size": np.max(chunk_lengths),
            "chunk_size_25": np.percentile(chunk_lengths, 25),
            "chunk_size_75": np.percentile(chunk_lengths, 75),
            "method": self.config.method.value,
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "split_by": self.config.split_by
            }
        }