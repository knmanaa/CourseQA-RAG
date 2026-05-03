## Plan: MATH Query Generation Pack

Build a MATH3424 query benchmark that mirrors the ECON/COMP/HUMA structure while honoring the 11-chapter scope and the 5-per-chapter-per-type quota, plus 7 special queries with feature codes 1..7. Use the same six query types and the 3-bit feature encoding with German as the multilingual bit.

**Steps**
1. Build the MATH3424 chapter map from [data/raw/MATH3424.json](data/raw/MATH3424.json) and [data/processed/corpus.jsonl](data/processed/corpus.jsonl), reconciling the requested 11 chapters with the actual lecture units and corpus IDs. Identify which corpus IDs represent each chapter and confirm there are enough non-boilerplate entries for 5 queries per chapter. *blocks step 2*
2. Define the query schema and ID policy by mirroring the existing outputs in [data/processed/queries/ECON/type_1/queries.jsonl](data/processed/queries/ECON/type_1/queries.jsonl). Confirm ID pattern (e.g., MATH_{type}_{nnn}) and metadata fields as strings. *blocks step 3*
3. Create a MATH worksheet CSV aligned to the column structure in [data/processed/queries/HUMA/huma_query_worksheet.csv](data/processed/queries/HUMA/huma_query_worksheet.csv). Populate 55 normal rows per type (11 chapters x 5, feature 0) and 7 special rows per type (feature 1..7). Assign corpus IDs for answerable rows and mark unanswerable ones. *blocks step 4*
4. Draft the per-type query text using chapter cues and the six-type taxonomy from [plan/HUMA_query_generation_instruction.md](plan/HUMA_query_generation_instruction.md). Apply the 3-bit feature scheme: bit2=German, bit1=noise, bit0=unanswerable, so codes 1,3,5,7 are unanswerable and must be excluded from ans.tsv. *blocks step 5*
5. Generate per-type outputs that match the existing BEIR-style structure in [data/processed/queries/ECON/type_1/queries.jsonl](data/processed/queries/ECON/type_1/queries.jsonl) and its paired ans.tsv. Include all queries in queries.jsonl and only answerable ones in ans.tsv. *depends on step 4*
6. Validate counts and distributions: 62 total per type (55 normal + 7 special), 5 normal queries per chapter, and exactly one special query for each feature 1..7. Verify ans.tsv references only existing corpus IDs from [data/processed/corpus.jsonl](data/processed/corpus.jsonl). *depends on step 5*
7. Spot-check 2 to 3 queries per type against the chapter source text to confirm type purity, German/noise correctness, and unanswerable exclusion from ans.tsv. *depends on step 5*

**Relevant files**
- [data/raw/MATH3424.json](data/raw/MATH3424.json) — primary MATH source content
- [data/processed/corpus.jsonl](data/processed/corpus.jsonl) — canonical corpus IDs used for qrels
- [plan/HUMA_query_generation_instruction.md](plan/HUMA_query_generation_instruction.md) — query type definitions and workflow template
- [data/processed/queries/HUMA/huma_query_worksheet.csv](data/processed/queries/HUMA/huma_query_worksheet.csv) — worksheet schema reference
- [data/processed/queries/ECON/type_1/queries.jsonl](data/processed/queries/ECON/type_1/queries.jsonl) — output format reference

**Verification**
1. Each type has 62 queries total, with 55 feature-0 normals and 7 specials (features 1..7).
2. Each of the 11 chapters contributes exactly 5 normal queries per type.
3. Feature distribution per type is exactly one query for each code 1..7; codes 1,3,5,7 are unanswerable and appear only in queries.jsonl.
4. Every ans.tsv query-id exists in queries.jsonl, and every corpus-id exists in [data/processed/corpus.jsonl](data/processed/corpus.jsonl).
5. No near-duplicate queries within a type, and special queries remain semantically aligned to their base intent (except unanswerables).

**Decisions**
- Use the same six query types as ECON/COMP/HUMA.
- Use 11 chapters with 5 normal queries per chapter per type (55 normals).
- Add 7 special queries per type with feature codes 1..7 only.
- Feature encoding uses 3 bits: bit2=German, bit1=noise, bit0=unanswerable.

**Further Considerations**
1. If the chapter map yields fewer than 11 usable chapters after filtering, decide whether to relax the 5-per-chapter rule or to reassign queries to adjacent chapters.
2. Confirm whether German queries should be fully German or a bilingual mix for consistency with prior multilingual examples.
3. If the processed corpus is sparse for certain chapters, consider allowing a second-best chapter cue while keeping the per-chapter counts stable.