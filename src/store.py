import os
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

FAISS_DIR = os.path.join("data", "faiss_store")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index.faiss")

def get_document_store() -> FAISSDocumentStore:
    """Returns the FAISSDocumentStore. Loads it if it exists on disk, otherwise creates it."""
    store = FAISSDocumentStore(embedding_dim=384)
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        store.load(index_path=FAISS_INDEX_PATH)
    else:
        print("Creating new FAISS document store")
        os.makedirs(FAISS_DIR, exist_ok=True)
    return store

def save_document_store(store: FAISSDocumentStore):
    """Saves the FAISSDocumentStore to disk."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    store.save(index_path=FAISS_INDEX_PATH)
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
