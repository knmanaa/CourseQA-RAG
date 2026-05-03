#!/usr/bin/env python3
"""Generate MATH query files (queries.jsonl + ans.tsv) from math_query_worksheet.csv."""

import csv
import json
import os

WORKSHEET_PATH = "data/processed/queries/MATH/math_query_worksheet.csv"
OUTPUT_DIR = "data/processed/queries/MATH"

# Read the worksheet
rows = []
with open(WORKSHEET_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Group by type
by_type = {}
for row in rows:
    t = int(row["type"])
    by_type.setdefault(t, []).append(row)

for t in range(1, 7):
    type_rows = by_type[t]
    type_dir = os.path.join(OUTPUT_DIR, f"type_{t}")
    os.makedirs(type_dir, exist_ok=True)

    queries = []
    ans_lines = []

    for row in type_rows:
        qid = row["query_id"].strip()
        text = row["text"].strip()
        feature = int(row["feature"])
        corpus_ids = row["corpus_ids"].strip()
        answerable = row["answerable"].strip().lower() == "yes"

        # Build query JSON
        query_obj = {
            "_id": qid,
            "text": text,
            "metadata": {
                "feature": str(feature),
                "type": str(t)
            }
        }
        queries.append(query_obj)

        # Build ans.tsv line if answerable
        if answerable and corpus_ids:
            ans_lines.append(f"{qid}\t{corpus_ids}\t1")

    # Write queries.jsonl
    q_path = os.path.join(type_dir, "queries.jsonl")
    with open(q_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write ans.tsv
    a_path = os.path.join(type_dir, "ans.tsv")
    with open(a_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for line in ans_lines:
            f.write(line + "\n")

    # Print stats
    normal = sum(1 for r in type_rows if int(r["feature"]) == 0)
    special = sum(1 for r in type_rows if int(r["feature"]) != 0)
    answerable_count = sum(1 for r in type_rows if r["answerable"].strip().lower() == "yes")
    print(f"Type {t}: {len(queries)} total queries ({normal} normal, {special} special), "
          f"{answerable_count} answerable, {len(ans_lines)} ans.tsv entries")

print("\nDone! All 6 types generated.")
