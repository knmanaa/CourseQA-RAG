## Plan: HUMA Query Generation Pack

Create a complete HUMA benchmark pack for six query types with strict schema and feature encoding, while keeping relevance labels compatible with BEIR-style qrels. Because HUMA content is book-like and more complex than COMP, the process emphasizes manual curation and careful anchoring to specific lecture text.

**Targets**
- Lecture unit: each HUMA1678 chapter (HUMA1678_chX in [data/raw/HUMA1678.json](data/raw/HUMA1678.json)).
- Current lecture count: 12 chapters (HUMA1678_ch1..ch12).
- Per type: 2 normal queries per lecture (feature code 0) plus 8 special queries (feature codes 0..7 exactly once each).
- Output format: queries.jsonl with {_id, text, metadata} and ans.tsv with header query-id, corpus-id, score.
- Unanswerable queries appear only in queries.jsonl and are excluded from ans.tsv.

**Steps**
1. Build lecture coverage map from [data/raw/HUMA1678.json](data/raw/HUMA1678.json) and/or [data/processed/corpus.jsonl](data/processed/corpus.jsonl). Filter boilerplate (ISBN, copyright, front matter) and keep substantive lecture text. *blocks step 2*
2. Draft a worksheet of candidate queries (one row per query) under [data/processed/queries/HUMA](data/processed/queries/HUMA). For each type and lecture, select two corpus ids and extract short cues for feature-0 queries. Add seven special rows per type with features 1..7 and mark answerable vs unanswerable. *blocks step 3*
	- Special features are encoded as 3-bit flags: bit2=multi-language (Vietnamese), bit1=noise (short-form and typo), bit0=unanswerable. Feature codes 0..7 represent all combinations.
3. Manual curation pass for clarity, type purity, and answerability. Ensure short paraphrases (no long quotes). Assign corpus ids for answerable specials. *blocks step 4*
4. Write output files per type: [data/processed/queries/HUMA/type_1/queries.jsonl](data/processed/queries/HUMA/type_1/queries.jsonl) through [data/processed/queries/HUMA/type_6/queries.jsonl](data/processed/queries/HUMA/type_6/queries.jsonl), plus matching ans.tsv files. Exclude unanswerables from ans.tsv. *depends on step 3*
5. Validate structure and labels: per type query count = 2 x 12 + 8 = 32; feature histogram per type is 0: 25 and 1..7: 1 each; query ids unique; ans.tsv query ids exist in queries; corpus ids exist in processed corpus. *depends on step 4*
6. Spot-check 2 to 3 queries per type against [data/raw/lectures/HUMA1678](data/raw/lectures/HUMA1678) or [data/processed/corpus.jsonl](data/processed/corpus.jsonl) for correctness and clarity. *depends on step 4*

**Schema and IDs**
- Query schema: {_id, text, metadata}, where metadata contains feature (string) and type (string).
- Type mapping: 1 factual, 2 hard-negative, 3 causal, 4 how-to, 5 multi-constraint, 6 hypothetical.
- Use consistent query ids (for example, HUMA_{type}_{nnn}) and keep them unique per type.

**Verification**
1. Counts: each type has 2 x 12 + 8 = 32 total queries; ans.tsv has header plus answerable rows only.
2. Feature coverage: feature 0..7 appears exactly once per type in special queries; remaining queries are feature 0.
3. Referential integrity: every ans.tsv query id exists in queries.jsonl; every corpus id exists in [data/processed/corpus.jsonl](data/processed/corpus.jsonl).
4. Quality: no near-duplicates within a type; unanswerables are truly unsupported by HUMA content.

**Decisions**
- Keep the ECON/COMP schema and six-type taxonomy.
- Use manual curation to handle HUMA complexity.
- Mix answerable and unanswerable special queries per type; unanswerables are excluded from ans.tsv.

**Further Considerations**
1. If a lecture chapter is mostly front matter, pull cues from the next substantive sections but keep the two-queries-per-lecture count stable.
2. If the effective lecture count changes after filtering, update the per-type targets and validation checks accordingly.
3. If you prefer all special queries to be answerable or all unanswerable, decide before drafting the worksheet.
