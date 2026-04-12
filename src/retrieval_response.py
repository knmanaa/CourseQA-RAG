import sys
import logging
import os
import urllib.request
from huggingface_hub import hf_hub_download

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

# Sử dụng Local Llama.cpp model
from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator

from src.store import get_document_store

print("✅ Loading Document Store (FAISS)...")
document_store = get_document_store()

if document_store.count_documents() == 0:
    print("⚠️ Document store is empty! Please run `python -m src.indexing` first.")
    sys.exit(1)
else:
    print(f"✅ Found {document_store.count_documents()} chunks in document store.")

# ====================== DOWNLOAD MODEL THUỘC HUB KNMANAA/COURSEQA ======================
print("✅ Checking Local GGUF Model (knmanaa/courseQA)...")
repo_id = "knmanaa/courseQA"
filename = "qwen3_5-4b-q4_K_M.gguf"  # quick test

# This function will download the model to the cache or return the path if already downloaded
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"✅ Model is ready at: {model_path}")

# ====================== PIPELINE ======================
rag_pipeline = Pipeline()

rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)

rag_pipeline.add_component(
    "retriever",
    FAISSEmbeddingRetriever(document_store=document_store, top_k=6)
)

rag_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=[
        ChatMessage.from_system(
            """You are a friendly AI assistant, only answer based on the context below.
            If the context does not contain the information to answer, say "I don't have enough information"."""),
        ChatMessage.from_user(
            """Context:\n{% for doc in documents %}{{ doc.content | truncate(500) }}\n{% endfor %}
Question: {{query}}"""
        )
    ])
)

rag_pipeline.add_component(
    "generator",
    # n_ctx is Context length, can be increased to 4096 if RAM is strong
    LlamaCppChatGenerator(model=model_path, n_ctx=2048, generation_kwargs={"max_tokens": 512, "temperature": 0.1})
)

# Connections
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")

print("✅ Pipeline Retrieval & Response (Local GGUF) is ready!")

# Test function
def run_query(question: str):
    result = rag_pipeline.run({
        "query_embedder": {"text": question},
        "prompt_builder": {"query": question}
    })
    return result["generator"]["replies"][0].text

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to Course RAG QA. Type 'exit' to quit.")
    print("="*50)
    
    while True:
        q = input("\n[You]: ")
        if q.strip().lower() in ['exit', 'quit']:
            break
            
        try:
            print("\n[AI is thinking...]")
            ans = run_query(q)
            # Remove <think> tags if present to just show the answer
            if "</think>" in ans:
                ans = ans.split("</think>")[-1].strip()
            print(f"\n[AI]: {ans}")
        except Exception as e:
            print(f"[Error]: {e}")
