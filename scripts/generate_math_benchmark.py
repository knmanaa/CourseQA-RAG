#!/usr/bin/env python3
"""
Generate MATH3424 query benchmark (queries.jsonl + ans.tsv) for all 6 types.

Each type:
  - 55 normal queries (11 chapters × 5, feature=0, all answerable)
  -  7 special queries (features 1..7, answerable per 3-bit encoding)
  - 62 total queries per type

Feature encoding (3-bit):
  bit2 = German (multilingual)
  bit1 = noise (typos / shorthand)
  bit0 = unanswerable
  Codes 1,3,5,7 → unanswerable (excluded from ans.tsv)
  Codes 0,2,4,6 → answerable

Output layout (mirrors ECON/COMP/HUMA):
  data/processed/queries/MATH/type_{1..6}/queries.jsonl
  data/processed/queries/MATH/type_{1..6}/ans.tsv
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
OUTPUT_ROOT = ROOT / "data" / "processed" / "queries" / "MATH"

TYPE_NAMES = {
    1: "factual",
    2: "hard-negative",
    3: "causal",
    4: "how-to",
    5: "multi-constraint",
    6: "hypothetical",
}

# ── Normal templates (5 per chapter per type) ──────────────────────────
# Each type has 5 distinct templates. {course}, {chapter}, {cue} are filled in.

NORMAL_TEMPLATES: Dict[int, List[str]] = {
    # ── Type 1: Factual ────────────────────────────────────────────────
    1: [
        "In {course} ch{chapter}, what is the definition of {cue}?",
        "According to {course} ch{chapter}, what does the lecture say about {cue}?",
        "From {course} ch{chapter}, what key formula or rule is presented for {cue}?",
        "What example does {course} ch{chapter} use to illustrate {cue}?",
        "What main takeaway should a student remember about {cue} from {course} ch{chapter}?",
    ],
    # ── Type 2: Hard-negative ──────────────────────────────────────────
    2: [
        "Does {course} ch{chapter} claim that {cue} has no effect on the outcome?",
        "Is it true that {cue} is always irrelevant according to {course} ch{chapter}?",
        "Did {course} ch{chapter} state the exact opposite of what is said about {cue}?",
        "Does the lecture in {course} ch{chapter} present {cue} as unnecessary for the analysis?",
        "Is {cue} described as optional rather than essential in {course} ch{chapter}?",
    ],
    # ── Type 3: Causal ─────────────────────────────────────────────────
    3: [
        "Why does {course} ch{chapter} say that {cue} matters for the regression result?",
        "What causal mechanism does the lecture draw between {cue} and the final outcome?",
        "Why does changing {cue} alter the behavior described in {course} ch{chapter}?",
        "What explanation does {course} ch{chapter} give for the effect of {cue}?",
        "Why does {cue} lead to the consequence shown in the example from {course} ch{chapter}?",
    ],
    # ── Type 4: How-to ─────────────────────────────────────────────────
    4: [
        "How do you apply the procedure for {cue} described in {course} ch{chapter}?",
        "How should a student work through the example about {cue} in {course} ch{chapter}?",
        "How do you check whether the condition for {cue} is satisfied in {course} ch{chapter}?",
        "How do you use the formula or method for {cue} to solve a problem in {course} ch{chapter}?",
        "How can you follow the step-by-step approach shown for {cue} in {course} ch{chapter}?",
    ],
    # ── Type 5: Multi-constraint ───────────────────────────────────────
    5: [
        "If {cue} and another condition from the lecture both hold, what conclusion follows in {course} ch{chapter}?",
        "When {cue} is combined with a second constraint, what outcome should you expect per {course} ch{chapter}?",
        "How does {cue} interact with a second assumption discussed in {course} ch{chapter}?",
        "If the lecture's main condition and {cue} change together, what happens according to {course} ch{chapter}?",
        "What result follows when {cue} is evaluated under two constraints at once in {course} ch{chapter}?",
    ],
    # ── Type 6: Hypothetical ───────────────────────────────────────────
    6: [
        "If {cue} were reversed, what counterfactual result would {course} ch{chapter} suggest?",
        "Under a hypothetical change to {cue}, how would the outcome move per {course} ch{chapter}?",
        "If the chapter's condition around {cue} no longer held, what would {course} ch{chapter} predict?",
        "What would the lecture in {course} ch{chapter} predict if {cue} were altered?",
        "If {cue} changed in the opposite direction, what would the slides in {course} ch{chapter} imply?",
    ],
}

# ── Special templates (one per feature code 1..7) ──────────────────────
# Each feature code gets exactly 1 template. The 3-bit encoding determines
# the style: bit2=German, bit1=noise, bit0=unanswerable.

SPECIAL_TEMPLATES: Dict[int, str] = {
    # feature 1 (001): unanswerable only — asks for fabricated numeric answer
    1: "What exact numeric answer does {course} give for a made-up version of {cue}?",
    # feature 2 (010): noise only — shorthand / typos
    2: "{course} ch{chapter} wat does {cue} mean? giv short ans pls",
    # feature 3 (011): noise + unanswerable — asks for false specific claim with typos
    3: "{course} ch{chapter} did lecture say exact opposite of {cue} is always true? rly?",
    # feature 4 (100): German only — multilingual
    4: "Was sagt {course} in Kapitel {chapter} über {cue}? Bitte erklären Sie kurz.",
    # feature 5 (101): German + unanswerable — asks for precise future prediction in German
    5: "Was ist die genaue Vorhersage von {course} in Kapitel {chapter} für {cue} im Jahr 2030?",
    # feature 6 (110): German + noise — mixed language with shorthand
    6: "{course} ch{chapter} wat sagt der lecture über {cue}? kurze antwort pls",
    # feature 7 (111): German + noise + unanswerable — fully corrupted
    7: "{course} ch{chapter} kann der lecture exact vorhersage für {cue} in 2030 geben? rly?",
}

# Which feature codes are answerable (bit0=0 → answerable)
ANSWERABLE_FEATURES = {0, 2, 4, 6}


@dataclass(frozen=True)
class CorpusItem:
    corpus_id: str
    course: str
    chapter: int
    slide: int
    text: str


def load_corpus(path: Path) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    pattern = re.compile(r"^(MATH3424)_ch(\d+)_(\d+)$")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            match = pattern.match(payload["_id"])
            if not match:
                continue
            course, chapter_text, slide_text = match.groups()
            items.append(
                CorpusItem(
                    corpus_id=payload["_id"],
                    course=course,
                    chapter=int(chapter_text),
                    slide=int(slide_text),
                    text=payload.get("text", ""),
                )
            )
    return items


def clean_text(text: str) -> str:
    replacements = {
        "\uf06e": " ",
        "\uf06c": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u21d2": "=>",
        "\ufb01": "fi",
        "\ufb02": "fl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_cue(text: str) -> str:
    """Extract a short cue phrase from the first substantive line of text."""
    lines = [clean_text(line) for line in text.splitlines()]
    candidates: List[str] = []
    skip_exact = {
        "Regression Analysis",
        "Math 3424",
        "Dr. Dong XIA",
        "Dr. Dong XIA (\u590f\u51ac)",
        "Outline",
        "Chapter 1: Introduction and Prerequisite",
        "Chapter 2: Simple Linear Regression",
        "Chapter 3: Multiple Linear Regression",
        "Chapter 4: Regression Diagnostics",
        "Chapter 5: Qualitative Predictors",
        "Chapter 7: Correlated Errors and Collinearity",
        "Chapter 8: Variable Selection",
        "Chapter 9: Logistic Regression",
    }
    for line in lines:
        if not line:
            continue
        if line in skip_exact:
            continue
        if re.fullmatch(r"p\.\d+", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", line.strip()):
            continue
        # Skip lines that are just slide numbers like "1", "2", etc.
        if re.fullmatch(r"\d+\s*$", line):
            continue
        # Skip pure header lines
        if line.startswith("Chapter ") and ":" in line:
            continue
        if line.lower().startswith("example:"):
            line = line.split(":", 1)[1].strip()
        if line.lower().startswith("question:"):
            line = line.split(":", 1)[1].strip()
        if line.lower().startswith("goal:"):
            line = line.split(":", 1)[1].strip()
        if line:
            candidates.append(line)
    if not candidates:
        return "the lecture topic"
    cue = candidates[0]
    words = cue.split()
    if len(words) > 10:
        cue = " ".join(words[:10])
    return cue


def spread_items(items: Sequence[CorpusItem], count: int = 5) -> List[CorpusItem]:
    """Evenly spread picks across the sorted items."""
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[0]]
    picks: List[CorpusItem] = []
    last_index = len(items) - 1
    used: set[str] = set()
    for slot in range(count):
        index = round(slot * last_index / (count - 1))
        item = items[index]
        if item.corpus_id in used:
            continue
        picks.append(item)
        used.add(item.corpus_id)
    if len(picks) < count:
        for item in items:
            if item.corpus_id in used:
                continue
            picks.append(item)
            used.add(item.corpus_id)
            if len(picks) == count:
                break
    return picks[:count]


def group_by_chapter(items: Sequence[CorpusItem]) -> Dict[int, List[CorpusItem]]:
    """Group MATH3424 items by chapter number."""
    grouped: Dict[int, List[CorpusItem]] = {}
    for item in items:
        grouped.setdefault(item.chapter, []).append(item)
    for key in grouped:
        grouped[key].sort(key=lambda entry: entry.slide)
    return grouped


def make_query_text(type_num: int, course: str, chapter: int, cue: str, slot: int) -> str:
    return NORMAL_TEMPLATES[type_num][slot].format(course=course, chapter=chapter, cue=cue)


def make_special_text(feature: int, course: str, chapter: int, cue: str) -> str:
    return SPECIAL_TEMPLATES[feature].format(course=course, chapter=chapter, cue=cue)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_tsv(path: Path, rows: Sequence[Tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, corpus_id, score in rows:
            handle.write(f"{query_id}\t{corpus_id}\t{score}\n")


def generate() -> None:
    corpus = load_corpus(CORPUS_PATH)
    grouped = group_by_chapter(corpus)

    # Determine available chapters (sorted)
    available_chapters = sorted(grouped.keys())
    print(f"Found {len(available_chapters)} MATH chapters: {available_chapters}")

    # We need 11 chapters (ch1..ch11 ideally)
    target_chapters = list(range(1, 12))
    missing = [ch for ch in target_chapters if ch not in available_chapters]
    if missing:
        print(f"WARNING: Missing chapters {missing}, will use available ones")

    # Use only chapters that exist
    chapters_to_use = [ch for ch in target_chapters if ch in available_chapters]
    if len(chapters_to_use) < 11:
        # Pad with extra chapters if available
        extra = [ch for ch in available_chapters if ch not in chapters_to_use]
        chapters_to_use.extend(sorted(extra)[: 11 - len(chapters_to_use)])

    chapters_to_use = sorted(chapters_to_use)[:11]
    print(f"Using {len(chapters_to_use)} chapters: {chapters_to_use}")

    for type_num in range(1, 7):
        query_rows: List[dict] = []
        ans_rows: List[Tuple[str, str, int]] = []
        query_counter = 1

        # ── Special queries (features 1..7) ────────────────────────────
        for feature in range(1, 8):
            # Pick anchor chapter: cycle through available chapters
            anchor_ch = chapters_to_use[(feature - 1) % len(chapters_to_use)]
            chapter_items = grouped.get(anchor_ch, [])
            if chapter_items:
                cue = extract_cue(chapter_items[0].text)
            else:
                cue = f"MATH3424 chapter {anchor_ch}"

            query_id = f"MATH_{type_num}_{query_counter:03d}"
            query_rows.append(
                {
                    "_id": query_id,
                    "text": make_special_text(feature, "MATH3424", anchor_ch, cue),
                    "metadata": {"feature": str(feature), "type": str(type_num)},
                }
            )

            # Add to ans.tsv only if answerable (bit0=0)
            if feature in ANSWERABLE_FEATURES and chapter_items:
                ans_rows.append((query_id, chapter_items[0].corpus_id, 1))

            query_counter += 1

        # ── Normal queries (5 per chapter, feature=0) ──────────────────
        for chapter in chapters_to_use:
            chapter_items = grouped.get(chapter, [])
            if len(chapter_items) < 5:
                print(
                    f"WARNING: Chapter {chapter} has only {len(chapter_items)} items, "
                    f"will use what's available"
                )
            selected = spread_items(chapter_items, 5)
            for slot, item in enumerate(selected):
                cue = extract_cue(item.text)
                query_id = f"MATH_{type_num}_{query_counter:03d}"
                query_rows.append(
                    {
                        "_id": query_id,
                        "text": make_query_text(type_num, "MATH3424", chapter, cue, slot),
                        "metadata": {"feature": "0", "type": str(type_num)},
                    }
                )
                ans_rows.append((query_id, item.corpus_id, 1))
                query_counter += 1

        # ── Validation ─────────────────────────────────────────────────
        expected_total = 55 + 7  # 62
        if len(query_rows) != expected_total:
            raise RuntimeError(
                f"Type {type_num} generated {len(query_rows)} queries, expected {expected_total}"
            )

        # Feature histogram: 0 appears 55 times, 1..7 appear 1 each
        histogram = {str(index): 0 for index in range(8)}
        for row in query_rows:
            histogram[row["metadata"]["feature"]] += 1
        expected_hist = {"0": 55}
        for f in range(1, 8):
            expected_hist[str(f)] = 1
        if histogram != expected_hist:
            raise RuntimeError(
                f"Type {type_num} feature histogram mismatch: {histogram} != {expected_hist}"
            )

        # Answerable count: feature 0 (55) + features 2,4,6 (3) = 58 answerable
        expected_ans = 55 + 3  # 58
        if len(ans_rows) != expected_ans:
            raise RuntimeError(
                f"Type {type_num} generated {len(ans_rows)} ans rows, expected {expected_ans}"
            )

        # ── Write output ───────────────────────────────────────────────
        out_dir = OUTPUT_ROOT / f"type_{type_num}"
        write_jsonl(out_dir / "queries.jsonl", query_rows)
        write_tsv(out_dir / "ans.tsv", ans_rows)

        print(
            f"type_{type_num} ({TYPE_NAMES[type_num]}): "
            f"{len(query_rows)} queries ({55} normal + {7} special), "
            f"{len(ans_rows)} ans rows"
        )

    print("\nDone! All 6 types generated for MATH3424.")


if __name__ == "__main__":
    generate()
