# Query Generation Guide — COMP, ECON, HUMA, MATH

This document explains how query-answer pairs are created across all four subjects in the CourseQA-RAG benchmark. It covers the shared schema, the distinction between **normal queries** and **special queries**, the 3-bit feature encoding, the six query types, and subject-specific generation strategies.

---

## 1. Shared Schema

All subjects follow a **BEIR-style** format with two output files per query type:

### `queries.jsonl`

Each line is a JSON object with three fields:

```json
{"_id": "<SUBJECT>_<TYPE>_<NNN>", "text": "...", "metadata": {"feature": "<0-7>", "type": "<1-6>"}}
```

| Field | Description |
|---|---|
| `_id` | Unique query identifier. Pattern: `{SUBJECT}_{TYPE}_{NNN}` (e.g., `ECON_1_01`, `COMP_1_001`, `HUMA_1_001`, `MATH_1_001`) |
| `text` | The query text in natural language |
| `metadata.feature` | 3-bit feature code as a string (`"0"` through `"7"`) — see [Feature Encoding](#2-feature-encoding) below |
| `metadata.type` | Query type as a string (`"1"` through `"6"`) — see [Six Query Types](#3-six-query-types) below |

### `ans.tsv`

Tab-separated file with a header row and one relevance label per line:

```
query-id\tcorpus-id\tscore
```

| Column | Description |
|---|---|
| `query-id` | Matches `_id` from `queries.jsonl` |
| `corpus-id` | The ID of a relevant document in the processed corpus (`data/processed/corpus.jsonl`) |
| `score` | Always `1` (binary relevance) |

**Rules:**
- Only **answerable** queries appear in `ans.tsv`.
- **Unanswerable** queries exist in `queries.jsonl` but have **no row** in `ans.tsv`.
- A query may map to multiple corpus IDs (one row per relevant document).

---

## 2. Feature Encoding

Each query carries a **3-bit feature code** (0–7) that encodes three orthogonal modifications:

| Bit | Value | Meaning | Description |
|---|---|---|---|
| bit0 | 1 | **Unanswerable** | The query asks about something **not present** in the course corpus (fabricated topic, future prediction, external fact). Queries with bit0=1 are excluded from `ans.tsv`. |
| bit1 | 2 | **Noise** | The query contains **typos, shorthand, or informal language** (e.g., "wht", "pls", "u", missing punctuation). |
| bit2 | 4 | **Multilingual** | The query is written in a **language other than English**. The language varies by subject (see subject sections below). |

### Feature Code Table

| Code | bit2 (lang) | bit1 (noise) | bit0 (unans.) | Answerable? | Description |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | ✅ Yes | **Normal query** — clean English, answerable |
| 1 | 0 | 0 | 1 | ❌ No | Clean English, unanswerable topic |
| 2 | 0 | 1 | 0 | ✅ Yes | Noisy English, answerable |
| 3 | 0 | 1 | 1 | ❌ No | Noisy English, unanswerable topic |
| 4 | 1 | 0 | 0 | ✅ Yes | Clean multilingual, answerable |
| 5 | 1 | 0 | 1 | ❌ No | Clean multilingual, unanswerable topic |
| 6 | 1 | 1 | 0 | ✅ Yes | Noisy multilingual, answerable |
| 7 | 1 | 1 | 1 | ❌ No | Noisy multilingual, unanswerable topic |

**Key takeaway:** Feature codes **1, 3, 5, 7** are always unanswerable (bit0=1) and are **omitted from `ans.tsv`**. Feature codes **0, 2, 4, 6** are answerable and **included in `ans.tsv`**.

---

## 3. Six Query Types

Every subject uses the same six-type taxonomy. Each type tests a different retrieval capability:

| Type | Name | Description | Example Pattern |
|---|---|---|---|
| 1 | **Factual** | Direct fact or definition retrieval from the lecture | "What is the definition of X?" |
| 2 | **Hard-negative** | Plausible-sounding but incorrect premise; tests ability to reject false statements | "Does the lecture claim that X causes Y?" (when it doesn't) |
| 3 | **Causal** | Cause-and-effect reasoning grounded in lecture content | "Why does X lead to Y according to the lecture?" |
| 4 | **How-to** | Procedural or process-oriented questions | "How is X calculated / implemented?" |
| 5 | **Multi-constraint** | Requires satisfying two or more conditions simultaneously | "What is the result when X and Y both hold?" |
| 6 | **Hypothetical** | "What if" scenarios that may or may not be answerable from the material | "If X changed, what would happen to Y?" |

---

## 4. Normal Queries vs. Special Queries

### Normal Queries (feature = 0)

- **Purpose:** Test standard retrieval performance on clean, answerable questions.
- **Feature code:** Always `"0"` (no modifications).
- **Answerability:** Always answerable (always in `ans.tsv`).
- **Content:** Chapter-grounded — each query references a specific lecture chapter and topic.
- **Distribution:** Evenly spread across chapters (each chapter gets the same number of normal queries per type).

### Special Queries (feature = 1..7)

- **Purpose:** Test retrieval robustness against edge cases: unanswerable questions, noisy text, and multilingual input.
- **Feature code:** One query per feature code 1..7 (or two per code for COMP).
- **Answerability:** Varies by feature code (codes 1,3,5,7 are unanswerable; codes 2,4,6 are answerable).
- **Content:** May reference fabricated topics (for unanswerable variants) or rephrase a normal query with noise/multilingual modifications.
- **Distribution:** Added as a separate block on top of the chapter-balanced normal queries.

### Summary Table

| Aspect | Normal Queries | Special Queries |
|---|---|---|
| Feature code | 0 | 1–7 |
| Answerable? | Always yes | Depends on bit0 |
| Chapter-grounded? | Yes | No (or loosely) |
| Language | Clean English | May have noise / other language |
| Purpose | Standard retrieval | Edge-case robustness |

---

## 5. Subject-Specific Details

### 5.1 ECON (ECON1220 — Economics)

| Aspect | Detail |
|---|---|
| **Source** | `data/raw/ECON1220.json` — lecture slides |
| **Chapters** | 11 (ch1–ch11) |
| **Queries per type** | **53 total** = 46 normal + 7 special |
| **Normal distribution** | 3–4 per chapter (not uniform: ch1, ch4, ch7 get extra queries) |
| **Special distribution** | Features 1..7 exactly once per type |
| **Multilingual language** | **Chinese** (Cantonese / Mandarin) |
| **Query ID pattern** | `ECON_{type}_{nn}` (e.g., `ECON_1_01`) |
| **Generation method** | Manual / semi-automated — queries are **chapter-grounded** using slide headings as cues (e.g., "According to ECON1220 ch1, what key point is made around '3. Rational People Think at the Margin'?") |
| **Key reference** | `plan/ECON_query_generation_instruction.md` |

**Example normal query (feature=0):**
```json
{"_id": "ECON_1_08", "text": "What does scarcity mean in this course?", "metadata": {"feature": "0", "type": "1"}}
```

**Example special queries:**
```json
{"_id": "ECON_1_02", "text": "wht is tax incidence in this course?", "metadata": {"feature": "2", "type": "1"}}
{"_id": "ECON_1_04", "text": "在這門課中，什麼是價格上限（price ceiling）？", "metadata": {"feature": "4", "type": "1"}}
```

---

### 5.2 COMP (COMP2011 + COMP2012 — Computer Science)

| Aspect | Detail |
|---|---|
| **Source** | `data/raw/COMP2011.json` + `data/raw/COMP2012.json` — lecture slides |
| **Chapters** | **30 total** (18 from COMP2011 + 12 from COMP2012) |
| **Queries per type** | **164 total** = 150 normal + 14 special |
| **Normal distribution** | **5 per chapter** (exactly 5 × 30 = 150) |
| **Special distribution** | **2 per feature code** 1..7 (2 × 7 = 14) |
| **Multilingual language** | N/A (COMP specials use fabricated/unanswerable topics rather than a second language) |
| **Query ID pattern** | `COMP_{type}_{nnn}` (e.g., `COMP_1_001`) |
| **Generation method** | **Template-based** — uses `NORMAL_TEMPLATES` and `SPECIAL_TEMPLATES` dictionaries in `scripts/generate_comp_benchmark.py`. Each template is a Python format string that gets filled with chapter/topic names. |
| **Key reference** | `plan/COMP_query_generation_instruction.md` |

**Example normal query (feature=0):**
```json
{"_id": "COMP_1_015", "text": "In COMP2011 ch1, what is the main idea of C++ Basics I?", "metadata": {"feature": "0", "type": "1"}}
```

**Example special queries:**
```json
{"_id": "COMP_1_001", "text": "What exact numeric answer does COMP2011 give for a made-up version of C++ Basics I?", "metadata": {"feature": "1", "type": "1"}}
{"_id": "COMP_1_003", "text": "COMP2011 ch1 what does C++ Basics I mean? Please explain briefly.", "metadata": {"feature": "2", "type": "1"}}
```

---

### 5.3 HUMA (HUMA1678 — Digital Humanities)

| Aspect | Detail |
|---|---|
| **Source** | `data/raw/HUMA1678.json` — textbook-style content |
| **Chapters** | 12 (ch1–ch12) |
| **Queries per type** | **32 total** = 24 normal + 8 special |
| **Normal distribution** | **2 per chapter** (exactly 2 × 12 = 24) |
| **Special distribution** | Features **0..7 exactly once** (includes feature 0 as an extra normal-style query) |
| **Multilingual language** | **Vietnamese** |
| **Query ID pattern** | `HUMA_{type}_{nnn}` (e.g., `HUMA_1_001`) |
| **Generation method** | **Worksheet-based manual curation** — uses a CSV worksheet (`data/processed/queries/HUMA/huma_query_worksheet.csv`) with columns: `query_id, type, lecture, feature, text, corpus_ids, answerable, notes`. Each row is manually drafted and reviewed. |
| **Key reference** | `plan/HUMA_query_generation_instruction.md` |

**Example normal query (feature=0):**
```json
{"_id": "HUMA_1_001", "text": "What does the text say about understanding digital humanities?", "metadata": {"feature": "0", "type": "1"}}
```

**Example special queries:**
```json
{"_id": "HUMA_1_026", "text": "What does the text say about quantum entanglement in Mars rovers?", "metadata": {"feature": "1", "type": "1"}}
{"_id": "HUMA_1_027", "text": "wht is understanding digital humanities as defined in te tx pls", "metadata": {"feature": "2", "type": "1"}}
{"_id": "HUMA_1_029", "text": "Theo bai doc, understanding digital humanities la gi?", "metadata": {"feature": "4", "type": "1"}}
```

---

### 5.4 MATH (MATH3424 — Regression Analysis)

| Aspect | Detail |
|---|---|
| **Source** | `data/raw/MATH3424.json` — lecture notes (11 chapters on Regression Analysis) |
| **Chapters** | 11 (ch1–ch11) |
| **Queries per type** | **62 total** = 55 normal + 7 special |
| **Normal distribution** | **5 per chapter** (exactly 5 × 11 = 55) |
| **Special distribution** | Features **1..7 exactly once** per type |
| **Multilingual language** | **German** |
| **Query ID pattern** | `MATH_{type}_{nnn}` (e.g., `MATH_1_001`) |
| **Generation method** | **Template-based** (similar to COMP) — uses `scripts/generate_math_benchmark.py` with `NORMAL_TEMPLATES` (5 per type) and `SPECIAL_TEMPLATES` (7 per type). Reads chapter structure from `MATH3424.json` and generates queries programmatically. |
| **Key reference** | `plan/MATH_query_generation_instruction.md` |

**Example normal query (feature=0):**
```json
{"_id": "MATH_1_008", "text": "In MATH3424 ch1, what is the definition of the lecture topic?", "metadata": {"feature": "0", "type": "1"}}
```

**Example special queries:**
```json
{"_id": "MATH_1_001", "text": "What exact numeric answer does MATH3424 give for a made-up version of the lecture topic?", "metadata": {"feature": "1", "type": "1"}}
{"_id": "MATH_1_002", "text": "MATH3424 ch2 wat does Chapter 2. Simple Linear Regression - Proofs mean? giv short ans pls", "metadata": {"feature": "2", "type": "1"}}
{"_id": "MATH_1_004", "text": "Was sagt MATH3424 in Kapitel 4 über Chapter 3. Multiple Linear Regression - Proofs? Bitte erklären Sie kurz.", "metadata": {"feature": "4", "type": "1"}}
```

---

## 6. Comparison Across Subjects

| Aspect | ECON | COMP | HUMA | MATH |
|---|---|---|---|---|
| **Course** | ECON1220 (Economics) | COMP2011+2012 (CS) | HUMA1678 (Digital Humanities) | MATH3424 (Regression Analysis) |
| **Chapters** | 11 | 30 (18+12) | 12 | 11 |
| **Queries/type** | 53 | 164 | 32 | 62 |
| **Normal/type** | 46 | 150 | 24 | 55 |
| **Special/type** | 7 | 14 | 8 | 7 |
| **Normal per chapter** | 3–4 (uneven) | 5 (uniform) | 2 (uniform) | 5 (uniform) |
| **Special coverage** | 1× features 1..7 | 2× features 1..7 | 1× features 0..7 | 1× features 1..7 |
| **Multilingual** | Chinese | — | Vietnamese | German |
| **Generation** | Manual / semi-auto | Template-based | Worksheet + manual | Template-based |
| **ID pattern** | `ECON_{t}_{nn}` | `COMP_{t}_{nnn}` | `HUMA_{t}_{nnn}` | `MATH_{t}_{nnn}` |

---

## 7. Directory Structure

```
data/processed/queries/
├── COMP/
│   ├── type_1/
│   │   ├── queries.jsonl
│   │   └── ans.tsv
│   ├── type_2/ ... type_6/
├── ECON/
│   ├── type_1/ ... type_6/
├── HUMA/
│   ├── huma_query_worksheet.csv
│   ├── type_1/ ... type_6/
└── MATH/
    ├── math_query_worksheet.csv
    ├── type_1/ ... type_6/
```

---

## 8. Validation Checklist

For any generated benchmark, verify:

1. **Counts:** Each type has the expected total (53/164/32/62), correct normal count, and correct special count.
2. **Feature distribution:** Feature 0 appears on all normal queries; features 1..7 appear exactly once (or twice for COMP) per type.
3. **Answerability:** Unanswerable queries (features 1,3,5,7) are present in `queries.jsonl` but absent from `ans.tsv`.
4. **ID uniqueness:** No duplicate `_id` values within or across types.
5. **Referential integrity:** Every `query-id` in `ans.tsv` exists in `queries.jsonl`; every `corpus-id` exists in `data/processed/corpus.jsonl`.
6. **Schema validity:** `queries.jsonl` is valid JSONL; `ans.tsv` has the correct header and tab separators.
