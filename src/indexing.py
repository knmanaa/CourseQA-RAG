import os
import json
from pathlib import Path
from typing import List, Union

from haystack import Pipeline, component
from haystack.dataclasses import Document, ByteStream
from haystack.components.converters import TextFileToDocument, PyPDFToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from src.chunker import DocumentChunker
from src.store import get_document_store, save_document_store
from src.ingest import get_test_files

@component
class CourseJSONToDocument:
    """A custom component to parse the scraped Course JSON (key=id, value=text) into Haystack Documents"""
    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        documents = []
        for src in sources:
            if isinstance(src, ByteStream):
                continue
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        # We create one document per slide/paragraph
                        documents.append(Document(content=val, meta={"slide_id": key, "file": str(src)}))
            except Exception as e:
                print(f"Warning: Could not read {src} as JSON - {e}")
        return {"documents": documents}

def run_indexing():
    files = get_test_files()
    print(f"Start indexing {len(files)} files: {files}")

    # ====================== INIT COMPONENTS ======================
    document_store = get_document_store()
    
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "application/json"])
    text_converter = TextFileToDocument()
    pdf_converter = PyPDFToDocument()
    json_converter = CourseJSONToDocument()
    
    document_joiner = DocumentJoiner()
    
    chunker = DocumentChunker(strategy="sentence", chunk_size=512, overlap=64)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    
    writer = DocumentWriter(document_store=document_store)
    
    # ====================== BUILD PIPELINE ======================
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    indexing_pipeline.add_component(instance=text_converter, name="text_converter")
    indexing_pipeline.add_component(instance=pdf_converter, name="pdf_converter")
    indexing_pipeline.add_component(instance=json_converter, name="json_converter")
    indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    indexing_pipeline.add_component(instance=chunker, name="chunker")
    indexing_pipeline.add_component(instance=embedder, name="embedder")
    indexing_pipeline.add_component(instance=writer, name="writer")
    
    # ====================== CONNECT ======================
    indexing_pipeline.connect("file_type_router.text/plain", "text_converter.sources")
    indexing_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
    indexing_pipeline.connect("file_type_router.application/json", "json_converter.sources")
    
    indexing_pipeline.connect("text_converter", "document_joiner")
    indexing_pipeline.connect("pdf_converter", "document_joiner")
    indexing_pipeline.connect("json_converter", "document_joiner")
    
    indexing_pipeline.connect("document_joiner", "chunker")
    indexing_pipeline.connect("chunker", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    # ====================== EXECUTE ======================
    indexing_pipeline.run({"file_type_router": {"sources": files}})
    
    save_document_store(document_store)
    print("✅ Indexing done & FAISS Store saved!")

if __name__ == "__main__":
    run_indexing()