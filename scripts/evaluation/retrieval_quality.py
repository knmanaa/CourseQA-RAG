from typing import Dict
from pathlib import Path
from torch import cuda
from datetime import datetime
import json
import sys 
import argparse

PROJECT_DIR = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_DIR)
from src.store import get_document_store
from src.util import timer

from beir.retrieval.evaluation import EvaluateRetrieval

from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.faiss.embedding_retriever import FAISSEmbeddingRetriever

@timer
def load_qrels(tsv_path: str) -> Dict[str, Dict[str, int]]:
    """
    Parse ans.tsv (query-id<TAB>corpus-id<TAB>score) into BEIR qrels format:
        {query_id: {corpus_id: relevance_score}}
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(tsv_path) as f:
        header = next(f)  # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, cid, score = line.split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels


@timer
def run_retrieval(queries_path: str, top_k: int):
    """
    Run  existing Haystack/FAISS retriever on every query in queries.jsonl.
    Returns a json file in BEIR results format:  {query_id: {corpus_id: score}}
    """

    # read queries from a location
    queries: Dict[str, str] = {}
    with open(queries_path) as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]


    # Prepare Pipeline
    device_str = "cuda:0" if cuda.is_available() else "cpu"
    device = ComponentDevice.from_str(device_str)

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(
        model = "deploy/embeddings/all-MiniLM-L6-v2",
        device = device
    ))
    pipeline.add_component("retriever", FAISSEmbeddingRetriever(
        document_store = get_document_store(),
        top_k = top_k
    ))
    pipeline.connect("embedder", "retriever")


    # Iterate all questions 
    query_retrieved_dict: Dict[str, Dict[str, float]] = {}
    for q_id, q_text in queries.items():
        run_input = {
            "embedder": {"text": q_text}
        }
        result = pipeline.run(run_input)
        docs = result.get("retriever", {}).get("documents", [])
        retrieved_docs_processed: Dict[str, str] = {}
        for doc in docs:
            clean_data_id = doc.id.split("_chunk_")[0]
            # if encounter same ID with lower score, omit.
            if (clean_data_id in retrieved_docs_processed.keys()) and doc.score < retrieved_docs_processed[clean_data_id]: continue
            retrieved_docs_processed[clean_data_id] = doc.score
        query_retrieved_dict[q_id] = retrieved_docs_processed
    

    # dump the dictionary into a json file
    target_file_dir = Path(PROJECT_DIR) / "log"
    with open(Path(target_file_dir) / "queries_retrieved_docs.json", "w") as f:
        json.dump(query_retrieved_dict, f, indent = 4)

    return query_retrieved_dict

@timer
def evaluate_FAISS_retrieval(args):
    top_k_values = [int(val.strip()) for val in args.top_k.split(",")]

    qrels = load_qrels(args.qrels)
    results = run_retrieval(args.queries, max(top_k_values))
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, top_k_values)
    return metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--corpus", default=str(Path(PROJECT_DIR) / "data" / "processed" /"corpus.jsonl"))
    parser.add_argument("--top-k", default="1,3,5,10,20,50")
    args = parser.parse_args()

    result_dict = evaluate_FAISS_retrieval(args) # big dictionary with results.
    
    params = {
        "queries": args.queries,
        "qrels": args.qrels,
        "corpus": args.corpus,
        "top-k": args.top_k
    }
    result = {"exec_time": str(datetime.now()), "args": params, "result": result_dict}
    with open(str(Path(PROJECT_DIR) / "log" / "retrieval_quality.jsonl"), "a") as f:
        json.dump(result, f)
        f.write("\n")
