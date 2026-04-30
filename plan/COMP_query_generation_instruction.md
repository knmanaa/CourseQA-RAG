## Plan: COMP Query Benchmark Expansion

Build the COMP benchmark to mirror the ECON format while scaling it to COMP2011 and COMP2012. The target is 150 chapter-based normal queries per type plus 14 special queries per type, with special coverage encoded as two queries for each feature code 1..7. The benchmark should stay append-only where practical, preserve the previous JSONL/TSV layout, and keep query IDs and qrels easy to validate.

**Steps**
1. Confirm COMP source coverage and chapter map.
1. Verify [data/raw/COMP2011.json](data/raw/COMP2011.json) and [data/raw/COMP2012.json](data/raw/COMP2012.json) contain the expected chapter ranges.
2. Map the 18 COMP2011 chapters and 12 COMP2012 chapters to a single 30-chapter planning grid.
3. Check the processed corpus for matching COMP corpus IDs before generating qrels.
2. Define the query schema and feature encoding.
1. Keep the previous BEIR-style structure: queries.jsonl plus ans.tsv with query-id, corpus-id, score.
2. Use feature 0 for the 150 normal chapter queries.
3. Use feature codes 1..7 twice each for the 14 special queries per type.
4. Keep type labels aligned with the existing six-type taxonomy already used in ECON.
3. Generate the per-type chapter-balanced query sets.
1. For each type, create 5 normal queries per chapter across all 30 chapters, for 150 normal queries total.
2. Keep query wording chapter-grounded so the same chapter does not get near-duplicate prompts across all 5 slots.
3. Let chapter difficulty vary slightly when a chapter is sparse or table-heavy, but preserve the 5-per-chapter target wherever possible.
4. Add the 14 special queries per type.
1. Create exactly two queries for each feature code 1..7.
2. Keep them distinct from the normal chapter queries so they stress different retrieval behavior.
3. Include the special queries as an additional block, not as replacements for chapter-balanced queries.
5. Write output files in the same folder structure as ECON.
1. Populate [data/processed/queries/COMP/type_1/queries.jsonl](data/processed/queries/COMP/type_1/queries.jsonl) through [data/processed/queries/COMP/type_6/queries.jsonl](data/processed/queries/COMP/type_6/queries.jsonl).
2. Populate matching ans.tsv files for answerable queries only.
3. Preserve any existing COMP folder layout and avoid renaming paths.
6. Validate the benchmark structure and label integrity.
1. Check that each type reaches 164 total queries: 150 normal plus 14 special.
2. Verify 5 normal queries per chapter across the 30-chapter grid.
3. Verify the special-query distribution is 2 per feature code 1..7.
4. Confirm qrels reference only corpus IDs that exist in the processed corpus.
5. Keep unanswerable queries out of ans.tsv, as in the ECON format.
7. Produce a concise handoff summary.
1. Summarize counts by type, chapter, and feature code.
2. Call out any chapters that required relaxed wording because the source material was sparse or disordered.

**Relevant files**
- /shared/project/CourseQA-RAG/data/raw/COMP2011.json — source lecture content for COMP2011
- /shared/project/CourseQA-RAG/data/raw/COMP2012.json — source lecture content for COMP2012
- /shared/project/CourseQA-RAG/data/processed/corpus.jsonl — canonical processed corpus used for qrels
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_1/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_1/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_2/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_2/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_3/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_3/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_4/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_4/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_5/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_5/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_6/queries.jsonl — COMP query output target
- /shared/project/CourseQA-RAG/data/processed/queries/COMP/type_6/ans.tsv — COMP relevance labels target
- /shared/project/CourseQA-RAG/plan/query_generation_instruction.md — format and constraints to preserve

**Verification**
1. Count verification:
1. Each type has 164 queries total.
2. Each type has 150 normal queries and 14 special queries.
3. Coverage verification:
1. Each of the 30 chapters contributes 5 normal queries per type.
2. Special queries are split evenly across feature codes 1..7.
3. Label verification:
1. ans.tsv only includes answerable queries.
2. Every corpus-id in ans.tsv exists in the processed corpus.
3. Schema verification:
1. JSONL remains valid.
2. TSV keeps the query-id, corpus-id, score header.
3. IDs remain unique and append-only if existing COMP outputs are already present.

**Decisions**
- COMP is the active scope; ECON is the reference format, not the target of this plan.
- Special queries are additional to the 150 chapter-balanced normal queries.
- The special-query scheme is 2 queries for each feature code 1..7.
- The chapter-balanced target is 5 queries per chapter across 30 chapters.

**Further Considerations**
1. If some COMP chapters are sparse or very table-heavy, the wording can be relaxed slightly while preserving the 5-per-chapter quota.
2. If existing COMP query folders already contain partial files, the next step should check whether to append or rebuild before writing new qrels.
3. If you want, the next pass can also define the six query types more explicitly so generation stays consistent across both COMP courses.