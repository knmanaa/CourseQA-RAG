import sys
print("Python:", sys.executable)

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

print("✅ Haystack imported successfully!")

# ====================== PIPELINE ======================
rag_pipeline = Pipeline()

rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)

rag_pipeline.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=6)
)

rag_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=[
        ChatMessage.from_system(
            """You are a helpful assistant answering questions based ONLY on the provided context.
            If the context doesn't contain the answer, say "I don't have enough information"."""),
        ChatMessage.from_user(
            """Context:\n{% for doc in documents %}{{ doc.content | truncate(500) }}\n{% endfor %}
Question: {{query}}"""
        )
    ])
)

rag_pipeline.add_component(
    "generator",
    OpenAIChatGenerator(model="gpt-4o-mini")
)

# Connections
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")

print("✅ Pipeline built successfully!")

# Test function
def run_query(question: str):
    result = rag_pipeline.run({
        "query_embedder": {"text": question},
        "prompt_builder": {"query": question}
    })
    return result["generator"]["replies"][0].content

if __name__ == "__main__":
    print(run_query("What is Haystack?"))