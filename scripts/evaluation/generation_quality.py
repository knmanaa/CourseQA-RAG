from haystack import Pipeline
from pathlib import Path
from typing import Dict, List
import json
import argparse

import sys
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.retrieval_response import run_query
from src.util import timer

@timer
def generate_answer(queries_filedir: str, answers_path = str(Path(PROJECT_DIR) / "log" / "LLM_answer.json")):
    # exepcted the queries are in jsonl format
    target_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(queries_filedir, "r") as f:
        for line in f:
            q_id, q_text, metadata = json.loads(line).values()

            ans, docs = run_query(q_text)
            ans = ans.split("</think>\n\n")[-1]

            target_dict[q_id] = [ans, {"chunks": [doc.id for doc in docs]}]
    
    with open(answers_path, "w") as f:
        json.dump(target_dict, f)


@timer
def judge_answer(answers_path: str = str(PROJECT_DIR / "log" / "LLM_answer.json"), target_path: str = str(Path(PROJECT_DIR) / "log" / "generation_quality.json")):

    # helper function for reading one line in jsonl queries.
    def jsonl_read_particular_line(jsonl_filedir: str, line_no: int):
        with open(jsonl_filedir, 'r') as f:
            for i, line in enumerate(f):
                if i == line_no:
                    return json.loads(line)


    # prepare LLM judge via OpenAI-compatible API (LaozhangAPI)
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LAOZHANG_API_KEY"),
        base_url=os.getenv("LAOZHANG_API_BASE", "https://api.laozhang.ai/v1"),
    )
    judge_model = os.getenv("LAOZHANG_JUDGE_MODEL", "deepseek-v4-flash")

    # prepare answer ... stuffs
    with open(answers_path) as f:
        dt = json.load(f)
    
    scores = {"relevance": [], "correctness": [], "completeness": [], "overall": [], "comment": []}; count = 0
    for q_id, bundle in dt.items():
        if count % 10 == 0: print(f"Now Processing Number {count} Query!")
        ans, chunk_ids = bundle[0], bundle[1]["chunks"]

        # find back the query using query id
        q_subj, q_type, q_no = q_id.split("_")
        q_filedir = Path(PROJECT_DIR) / "data" / "processed" / "queries" / f"{q_subj}" / f"type_{q_type}" / "queries.jsonl"
        q_text = jsonl_read_particular_line(str(q_filedir), q_no)

        # find back the content of chunks using chunk ids.
        chunk_contents = []
        with open(str(PROJECT_DIR / "data" / "faiss_store" / "faiss_index.json"), "r") as f:
            index_json = json.load(f)
            for chunk_id in chunk_ids:
                chunk_contents.append(
                    index_json["documents"][index_json["inverse_id_map"][chunk_id]]["content"]
                )
        context_text = "\n\n".join(chunk_contents)



        prompt = f"""You are an expert evaluator. Rate this answer 1-5.

        Question: {q_text}

        Context provided:
        {context_text}

        Answer: {ans}

        Return JSON: {{"relevance": 1-5, "correctness": 1-5, "completeness": 1-5, "overall": 1-5, "comment": "One or two short sentence, explain the score you give."}} 
        
        Only output JSON, no other text."""

        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        try:
            rating = json.loads(resp.choices[0].message.content)
            for k in scores.keys():
                scores[k].append(rating.get(k, 0))
        except json.JSONDecodeError:
            print(f"  ⚠️  Judge parse failed for {q_id}")

        count += 1

    print("\n=== LLM-as-Judge Results ===")
    for metric, vals in scores.items():
        if metric == "comment": continue
        if vals:
            print(f"  {metric}: mean={sum(vals)/len(vals):.2f}")

    with open(target_path, "w") as f:
        json.dump(scores, f, indent = 4)

    return scores



@timer
def evaluate_LLM_generation(args):

    print("Now Generating Answer")
    generate_answer(args.queries, args.ans)
    print("Now Judging Answer")
    judge_answer(args.ans, args.target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ans", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    evaluate_LLM_generation(args)