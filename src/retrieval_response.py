import sys
import logging
import os
import inspect
from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError

# --- Windows compatibility fixes (must run before any network/SSL imports) ---
# 1. Force UTF-8 stdout/stderr so emoji in print() don't crash on cp950 terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 2. If SSL_CERT_FILE points to a file that doesn't exist (e.g. stale conda env
#    path set in the shell), unset it so httpx falls back to the system/certifi
#    certificate store rather than raising FileNotFoundError.
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    del os.environ["SSL_CERT_FILE"]
# ---------------------------------------------------------------------------

from huggingface_hub import hf_hub_download

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import ComponentDevice
import torch
import llama_cpp
from llama_cpp import Llama

# Sử dụng Local Llama.cpp model
from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator

from src.store import get_document_store


def _parse_semver(raw: str) -> tuple[int, int, int]:
    base = raw.split("+", 1)[0].split("-", 1)[0]
    parts = base.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _ensure_llama_cpp_compatibility() -> None:
    min_version = (0, 3, 20)
    try:
        current = _parse_semver(pkg_version("llama-cpp-python"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it in your active environment."
        ) from exc

    if current < min_version:
        raise RuntimeError(
            "llama-cpp-python is too old for Qwen3.5 GGUF models. "
            f"Found {current[0]}.{current[1]}.{current[2]}, need >= {min_version[0]}.{min_version[1]}.{min_version[2]}.\n"
            "Run: pip install --upgrade llama-cpp-python>=0.3.20"
        )


_ensure_llama_cpp_compatibility()


def _llama_supports_gpu_offload() -> bool:
    support_fn = getattr(llama_cpp.llama_cpp, "llama_supports_gpu_offload", None)
    if callable(support_fn):
        try:
            return bool(support_fn())
        except Exception:
            return False
    return False


def _ensure_llama_gpu_backend(requested_gpu_layers: int) -> None:
    if requested_gpu_layers == 0:
        return

    torch_has_cuda = torch.cuda.is_available()
    llama_has_gpu = _llama_supports_gpu_offload()
    print(f"[GPU CHECK] torch.cuda.is_available={torch_has_cuda}, llama_supports_gpu_offload={llama_has_gpu}")

    if torch_has_cuda and not llama_has_gpu:
        raise RuntimeError(
            "CUDA is available in torch, but llama-cpp-python in this environment was built without GPU offload support.\n"
            "Reinstall llama-cpp-python in CourseQARAG with CUDA support, for example:\n"
            "  /shared/conda_envs/CourseQARAG/bin/python3 -m pip uninstall -y llama-cpp-python\n"
            "  CMAKE_ARGS='-DGGML_CUDA=on' FORCE_CMAKE=1 "
            "/shared/conda_envs/CourseQARAG/bin/python3 -m pip install --no-binary=:all: llama-cpp-python==0.3.20\n"
            "Then rerun this script."
        )

print("✅ Loading Document Store (FAISS)...")
document_store = get_document_store()

if document_store.count_documents() == 0:
    print("⚠️ Document store is empty! Please run `python -m src.indexing` first.")
    sys.exit(1)
else:
    print(f"✅ Found {document_store.count_documents()} chunks in document store.")

def _is_valid_gguf(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size < 8:
        return False
    with p.open("rb") as f:
        return f.read(4) == b"GGUF"


# ====================== DOWNLOAD MODEL THUỘC HUB KNMANAA/COURSEQA ======================
print("✅ Checking Local GGUF Model (knmanaa/courseQA)...")
repo_id = os.getenv("COURSEQA_MODEL_REPO", "knmanaa/courseQA")
filename = os.getenv("COURSEQA_MODEL_FILE", "qwen3_5-4b-q4_K_M.gguf")

# Download from HF cache, then validate GGUF header to detect bad cache entries.
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
if not _is_valid_gguf(model_path):
    print("⚠️ Cached model appears invalid. Re-downloading...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=True)

if not _is_valid_gguf(model_path):
    raise RuntimeError(
        f"Model file is not a valid GGUF: {model_path}. "
        "Delete the cache entry and try again."
    )

print(f"✅ Model is ready at: {model_path}")

# ====================== PIPELINE ======================
rag_pipeline = Pipeline()

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
device = ComponentDevice.from_str(device_str)

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
device = ComponentDevice.from_str(device_str)

rag_pipeline.add_component(
    "query_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", device=device)
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", device=device)
)

rag_pipeline.add_component(
    "retriever",
    FAISSEmbeddingRetriever(document_store=document_store, top_k=6)
    FAISSEmbeddingRetriever(document_store=document_store, top_k=6)
)

rag_pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(template=[
        ChatMessage.from_system(
            """You are a helpful AI assistant.
Answer the exact user question without reinterpreting it.
Prefer the provided context when it is relevant.
If context is weak or missing for the asked question, provide a short best-effort answer from general knowledge and state that context was insufficient.
Do NOT show your thinking process. Do NOT use <think> tags. Do NOT output reasoning steps.
Output only the final answer directly. If information is missing, say "I don't have enough information"."""),
        ChatMessage.from_user(
            """Context:\n{% for doc in documents %}{{ doc.content | truncate(500) }}\n{% endfor %}
Question: {{query}}"""
        )
    ], required_variables=["query"])
)

preferred_gpu_layers = int(os.getenv("COURSEQA_N_GPU_LAYERS", "-1" if torch.cuda.is_available() else "0"))
supports_chat_template_kwargs = (
    "chat_template_kwargs" in inspect.signature(Llama.create_chat_completion).parameters
)

_ensure_llama_gpu_backend(preferred_gpu_layers)


def _build_generator(n_gpu_layers: int) -> LlamaCppChatGenerator:
    # n_ctx is context length, set to 4096 if enough RAM.
    # Keep max_tokens low to discourage thinking output e.g. 256
    generation_kwargs = {
        "max_tokens": 1024,
        "top_p": 0.85,
        "repeat_penalty": 1.1,
        "temperature": 0.1
    }
    if supports_chat_template_kwargs:
        generation_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    return LlamaCppChatGenerator(
        model=model_path,
        n_ctx=4096,
        model_kwargs={
            "n_gpu_layers": n_gpu_layers,
            "chat_format": "qwen"
        },
        generation_kwargs=generation_kwargs
    )


try:
    rag_pipeline.add_component("generator", _build_generator(preferred_gpu_layers))
except Exception as exc:
    if preferred_gpu_layers != 0:
        print(
            "⚠️ Failed to initialize GGUF model with GPU offload "
            f"(n_gpu_layers={preferred_gpu_layers}). Retrying on CPU..."
        )
        rag_pipeline.add_component("generator", _build_generator(0))
    else:
        raise RuntimeError(
            f"Failed to load model from file: {model_path}\n"
            "Try upgrading llama-cpp-python/llama-cpp-haystack or set COURSEQA_N_GPU_LAYERS=0."
        ) from exc

# Connections
rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
rag_pipeline.connect("prompt_builder.prompt", "generator.messages")

print("✅ Pipeline Retrieval & Response (Local GGUF) is ready!")
debug_retrieval = os.getenv("COURSEQA_DEBUG_RETRIEVAL", "0") == "1"
retrieval_min_score = float(os.getenv("COURSEQA_MIN_RETRIEVAL_SCORE", "0.25"))

# Test function
def run_query(question: str):
    run_input = {
        "query_embedder": {"text": question},
        "prompt_builder": {"query": question},
    }
    if debug_retrieval:
        result = rag_pipeline.run(run_input, include_outputs_from=["retriever"])
    else:
        result = rag_pipeline.run(run_input)

    docs = result.get("retriever", {}).get("documents", []) if debug_retrieval else []
    if debug_retrieval:
        print("\n[DEBUG] Top retrieved chunks:")
        for idx, doc in enumerate(docs, start=1):
            meta = doc.meta or {}
            src = meta.get("file") or meta.get("file_path") or meta.get("source") or "<unknown>"
            snippet = (doc.content or "")[:220].replace("\n", " ")
            print(f"  {idx}. score={doc.score:.4f} source={src}")
            print(f"     {snippet}")
        strong_docs = [d for d in docs if (d.score or 0.0) >= retrieval_min_score]
        print(
            f"[DEBUG] Retrieval strength: {len(strong_docs)}/{len(docs)} chunks above "
            f"score {retrieval_min_score:.2f}"
        )

    answer = result["generator"]["replies"][0].text
    
    return answer

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
