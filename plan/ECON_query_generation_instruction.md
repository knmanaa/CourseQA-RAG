## Plan: ECON Query Generation Pack

Create a complete ECON benchmark pack for six query types with strict schema and feature encoding, while keeping relevance labels compatible with BEIR-style qrels. The implementation produced 318 ECON queries total (53 per type). Each chapter contributed 3 normal, chapter-grounded queries (target: 11 chapters × 3 = 33 normal queries per type, adding 13 normal queries from chapter 1, 4 and 7). Feature-code coverage remains: codes 1..7 appear once per type and the remaining queries use feature code 0. Matching ans.tsv files contain positive labels only (unanswerable queries are omitted from ans.tsv).

**Steps**
1. Prepare source mapping and constraints from existing ECON corpus data. Build a doc map from [data/raw/ECON1220.json](data/raw/ECON1220.json) and/or [data/processed/corpus.jsonl](data/processed/corpus.jsonl), preserving stable corpus ids like ECON1220_chX_Y. *blocks all later steps*
2. Create output directories for six types under [data/processed/queries/ECON](data/processed/queries/ECON): type_1 through type_6, each containing queries jsonl and ans.tsv. *parallel with step 3 once paths are fixed*
3. Define deterministic ID and metadata policy. Use query ids ECON_{type}_{nn} for nn from 01 to 20; metadata contains feature and type as strings; type mapping fixed as 1 factual, 2 hard-negative, 3 causal, 4 how-to, 5 multi-constraint, 6 hypothetical. *blocks step 4*
4. Generate type 1 factual queries (53). Ensure 7 special queries carry feature codes 1..7 exactly once and the remaining 46 normal queries carry feature code 0. Anchor each answerable query to one or more ECON corpus ids; mark designated unanswerables by generating no ans row. *parallel with steps 5 to 9 conceptually, but can be executed sequentially for quality control*
5. Generate type 2 hard-negative queries (53) with strong lexical overlap and incorrect premises. Keep clear evidence mapping for answerable corrections; omit ans rows for chosen unanswerables. Apply the same feature distribution rule as step 4.
6. Generate type 3 causal queries (53). Ensure causal direction is grounded in slide content and avoid speculative external facts. Apply same feature distribution and ans rules.
7. Generate type 4 how-to queries (53). Restrict to procedural reasoning derivable from lecture material (for example, how to analyze tax incidence under elasticity assumptions). Apply same feature distribution and ans rules.
8. Generate type 5 multi-constraint queries (53). Require two or more constraints per query, and allow multiple corpus ids in ans.tsv where evidence spans slides. Apply same feature distribution and ans rules.
9. Generate type 6 hypothetical queries (53). Keep hypotheticals in-scope for ECON content; include deliberately unanswerable cases and omit their ans rows. Apply same feature distribution and ans rules.
10. Write files per type: queries jsonl with schema {_id, text, metadata} and ans.tsv with header query-id, corpus-id, score. Use tab-separated rows and score 1 for relevant pairs only. *depends on steps 4 to 9*
11. Run validation checks: schema validity, exactly 53 queries per type, exact feature histogram per type (0:46, 1..7:1 each), id uniqueness, ans query-id subset of queries ids, corpus-id existence in ECON corpus, and no malformed tabs/newlines. *depends on step 10*
12. Perform quality review pass for overlap control and type purity. Detect near-duplicates across all 120 ECON queries, verify hard-negative realism, and verify unanswerable items are truly unsupported by ECON corpus. *depends on step 11*

**Relevant files**
- /shared/project/CourseQA-RAG/data/raw/ECON1220.json — primary ECON content source at slide granularity
- /shared/project/CourseQA-RAG/data/processed/corpus.jsonl — canonical processed corpus ids used by qrels mapping
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_1 — factual outputs: queries and ans
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_2 — hard-negative outputs: queries and ans
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_3 — causal outputs: queries and ans
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_4 — how-to outputs: queries and ans
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_5 — multi-constraint outputs: queries and ans
- /shared/project/CourseQA-RAG/data/processed/queries/ECON/type_6 — hypothetical outputs: queries and ans
- /shared/project/CourseQA-RAG/dataset/datasets/scifact/qrels/train.tsv — reference for tsv header and positive-label convention
- /shared/project/CourseQA-RAG/dataset/datasets/scifact/qrels/test.tsv — reference for repeated query-id across multiple corpus ids

**Verification**
1. Count checks: each type file has exactly 53 query lines.
2. Feature checks per type: feature codes include exactly one each of 1,2,3,4,5,6,7 and the remaining 46 queries use feature code 0.
3. ID checks: ids follow ECON_{type}_{nn}, unique within type and globally unique across ECON set.
4. Answer checks: ans.tsv has header and only score 1 rows; no ans row for unanswerable queries.
5. Referential checks: every ans query-id exists in its queries file; every corpus-id exists in processed ECON corpus.
6. Multi-evidence checks: allow repeated query-id rows for multi-constraint evidence sets.
7. Quality checks: no near-duplicate query text across types beyond acceptable paraphrase noise cases.
8. Manual spot-check: at least 3 queries per type verified against source slides for label correctness.

**Decisions**
- Use six fixed types: 1 factual, 2 hard-negative, 3 causal, 4 how-to, 5 multi-constraint, 6 hypothetical.
- Per type, special-feature distribution is fixed to one query each for codes 1..7 and thirteen queries with code 0.
- Unanswerable queries appear in queries jsonl but are omitted from ans.tsv.
- ans.tsv includes header row query-id, corpus-id, score and uses tab separators.
- Scope for this delivery is ECON only.

**Further Considerations**
1. If a query can be answered by multiple slides, include all supporting corpus ids in ans.tsv to reward multi-document retrieval.
2. Keep multilingual and noisy variants semantically equivalent to their base intent, except where intentionally unanswerable.
3. For hard-negative items, ensure the false premise is close to course wording to stress retrieval and reasoning without becoming trivial.