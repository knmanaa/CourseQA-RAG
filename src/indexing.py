from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
# Or: from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator  # or HuggingFaceLocalGenerator, Ollama...
from haystack.dataclasses import ChatMessage

rag_pipeline = Pipeline()

# 1. Embed query
rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")  # needs to match the indexing
)

# 2. Retrieve
rag_pipeline.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=6)  # top_k 5-10 
)

# 3. (Optional) Reranker to filter context with good quality
# rag_pipeline.add_component("reranker", SentenceTransformersRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3))

# 4. Prompt builder (use chat format for better response)
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

# 5. Generator
rag_pipeline.add_component(
    "generator",
    OpenAIChatGenerator(model="gpt-4o-mini")  # substitute by local: HuggingFaceLocalGenerator(...) or Ollama
)

# Connections
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")
# Nếu có reranker: retriever → reranker → prompt_builder
def run_query(question: str):
    """Main function to run query"""
    result = rag_pipeline.run({
        "query_embedder": {"text": question},
        "prompt_builder": {"query": question}
    })
    return result["generator"]["replies"][0].content

# Quick test in this file (optional)
if __name__ == "__main__":
    print(run_query("what's your question?"))