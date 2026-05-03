#!/usr/bin/env python3
"""Generate the COMP benchmark query packs.

The output mirrors the ECON benchmark layout:
- six query types
- JSONL queries files
- TSV relevance files
- 150 normal queries per type
- 14 special queries per type
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
OUTPUT_ROOT = ROOT / "data" / "processed" / "queries" / "COMP"

TYPE_NAMES = {
    1: "factual",
    2: "hard-negative",
    3: "causal",
    4: "how-to",
    5: "multi-constraint",
    6: "hypothetical",
}

SPECIAL_FEATURE_SEQUENCE = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

NORMAL_TEMPLATES = {
    1: [
        "In {course} ch{chapter}, what is the main idea of {cue}?",
        "According to {course} ch{chapter}, how is {cue} introduced?",
        "What example does {course} ch{chapter} use for {cue}?",
        "Which rule or property is highlighted in {course} ch{chapter} around {cue}?",
        "What takeaway should a student remember from {cue} in {course} ch{chapter}?",
    ],
    2: [
        "Does {course} ch{chapter} claim that {cue} has no effect on the outcome?",
        "Is it true that {cue} works even when the lecture's usual condition is missing?",
        "Did {course} ch{chapter} say the opposite of {cue} is always true?",
        "Does the lecture present {cue} as irrelevant to the result?",
        "Is {cue} described as optional rather than necessary in {course} ch{chapter}?",
    ],
    3: [
        "Why does {course} ch{chapter} say {cue} matters?",
        "What causal link does the lecture draw between {cue} and the final outcome?",
        "Why does changing {cue} change the behavior described in the slide?",
        "What mechanism explains {cue} in {course} ch{chapter}?",
        "Why does {cue} lead to the effect shown in the example?",
    ],
    4: [
        "How do you apply the rule about {cue} in {course} ch{chapter}?",
        "How should you work through the example around {cue}?",
        "How do you check whether the condition in {cue} is satisfied?",
        "How do you use the lecture procedure for {cue} to solve a problem?",
        "How can you follow the steps shown for {cue}?",
    ],
    5: [
        "If {cue} and another condition both hold, what conclusion follows?",
        "When {cue} is combined with a second constraint, what outcome should you expect?",
        "How does {cue} interact with a second lecture assumption?",
        "If the lecture's condition and {cue} change together, what happens?",
        "What result follows when {cue} is evaluated under two constraints at once?",
    ],
    6: [
        "If {cue} were reversed, what counterfactual result would the lecture suggest?",
        "Under a hypothetical change to {cue}, how would the outcome move?",
        "If the chapter's condition around {cue} no longer held, what would happen?",
        "What would the lecture predict if {cue} were altered?",
        "If {cue} changed in the opposite direction, what would the slides imply?",
    ],
}

SPECIAL_TEMPLATES = {
    1: [
        "What exact numeric answer does {course} give for a made-up version of {cue}?",
        "Please state the precise number for {cue} in {course}, even if the lecture never gives one.",
    ],
    2: [
        "{course} ch{chapter} what does {cue} mean? Please explain briefly.",
        "{course} ch{chapter} says {cue}? The wording is messy, but give the short answer.",
    ],
    3: [
        "Does {course} ch{chapter} say that the opposite of {cue} is always correct?",
        "Did the lecture claim {cue} has no role at all in the result?",
    ],
    4: [
        "In {course} ch{chapter}, how is {cue} used?",
        "Use the content of {course} ch{chapter} to explain {cue}.",
    ],
    5: [
        "If {cue} and two extra constraints all have to hold, what exact answer would the lecture produce?",
        "When {cue} is mixed with another restriction and a numeric condition, what result should we get?",
    ],
    6: [
        "Can you explain {cue} from {course} ch{chapter} in plain English and code-like shorthand?",
        "{course} ch{chapter}, please give the {cue} idea with a short mix of English and symbols.",
    ],
    7: [
        "Would {course} guarantee a future outcome for {cue} if the conditions keep changing?",
        "If the future flips, what would {course} predict about {cue}?",
    ],
}


@dataclass(frozen=True)
class CorpusItem:
    corpus_id: str
    course: str
    chapter: int
    slide: int
    text: str


def load_corpus(path: Path) -> List[CorpusItem]:
    items: List[CorpusItem] = []
    pattern = re.compile(r"^(COMP201[12])_ch(\d+)_(\d+)$")
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
    lines = [clean_text(line) for line in text.splitlines()]
    candidates: List[str] = []
    skip_exact = {
        "Programming with C++",
        "Part I",
        "Part II",
        "Part III",
        "Part IV",
        "Part V",
        "Part VI",
        "Part VII",
        "Part VIII",
        "Part IX",
        "Department of Computer Science & Engineering",
        "Hong Kong SAR, China",
    }
    for line in lines:
        if not line:
            continue
        if line in skip_exact:
            continue
        if re.fullmatch(r"p\.\d+", line, flags=re.IGNORECASE):
            continue
        if line.startswith("COMP201") and ":" in line:
            line = line.split(":", 1)[1].strip()
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
    if len(words) > 8:
        cue = " ".join(words[:8])
    return cue


def spread_items(items: Sequence[CorpusItem], count: int = 5) -> List[CorpusItem]:
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


def group_by_chapter(items: Sequence[CorpusItem]) -> Dict[Tuple[str, int], List[CorpusItem]]:
    grouped: Dict[Tuple[str, int], List[CorpusItem]] = {}
    for item in items:
        grouped.setdefault((item.course, item.chapter), []).append(item)
    for key in grouped:
        grouped[key].sort(key=lambda entry: entry.slide)
    return grouped


def make_query_text(type_num: int, course: str, chapter: int, cue: str, slot: int) -> str:
    return NORMAL_TEMPLATES[type_num][slot].format(course=course, chapter=chapter, cue=cue)


def make_special_query(type_num: int, feature: int, ordinal: int, course: str, chapter: int, cue: str) -> str:
    return SPECIAL_TEMPLATES[feature][ordinal].format(course=course, chapter=chapter, cue=cue)


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


def chapter_range(course: str) -> range:
    if course == "COMP2011":
        return range(1, 19)
    if course == "COMP2012":
        return range(1, 13)
    raise ValueError(f"Unknown course {course}")


def generate() -> None:
    corpus = load_corpus(CORPUS_PATH)
    grouped = group_by_chapter(corpus)

    for type_num in range(1, 7):
        query_rows: List[dict] = []
        ans_rows: List[Tuple[str, str, int]] = []
        query_counter = 1

        for special_index, feature in enumerate(SPECIAL_FEATURE_SEQUENCE):
            ordinal = special_index % 2
            anchor_course = "COMP2011" if ordinal == 0 else "COMP2012"
            anchor_chapter = 1
            chapter_items = grouped.get((anchor_course, anchor_chapter), [])
            cue = extract_cue(chapter_items[0].text) if chapter_items else f"{anchor_course} chapter {anchor_chapter}"
            query_id = f"COMP_{type_num}_{query_counter:03d}"
            query_rows.append(
                {
                    "_id": query_id,
                    "text": make_special_query(type_num, feature, ordinal, anchor_course, anchor_chapter, cue),
                    "metadata": {"feature": str(feature), "type": str(type_num)},
                }
            )
            query_counter += 1

        for course in ("COMP2011", "COMP2012"):
            for chapter in chapter_range(course):
                chapter_items = grouped.get((course, chapter), [])
                selected = spread_items(chapter_items, 5)
                if len(selected) < 5:
                    raise RuntimeError(f"{course} chapter {chapter} does not have 5 corpus items")
                for slot, item in enumerate(selected):
                    cue = extract_cue(item.text)
                    query_id = f"COMP_{type_num}_{query_counter:03d}"
                    query_rows.append(
                        {
                            "_id": query_id,
                            "text": make_query_text(type_num, course, chapter, cue, slot),
                            "metadata": {"feature": "0", "type": str(type_num)},
                        }
                    )
                    ans_rows.append((query_id, item.corpus_id, 1))
                    query_counter += 1

        if len(query_rows) != 164:
            raise RuntimeError(f"Type {type_num} generated {len(query_rows)} queries, expected 164")
        if len(ans_rows) != 150:
            raise RuntimeError(f"Type {type_num} generated {len(ans_rows)} answer rows, expected 150")

        histogram = {str(index): 0 for index in range(8)}
        for row in query_rows:
            histogram[row["metadata"]["feature"]] += 1
        expected = {"0": 150, **{str(index): 2 for index in range(1, 8)}}
        if histogram != expected:
            raise RuntimeError(f"Type {type_num} feature histogram mismatch: {histogram}")

        out_dir = OUTPUT_ROOT / f"type_{type_num}"
        write_jsonl(out_dir / "queries.jsonl", query_rows)
        write_tsv(out_dir / "ans.tsv", ans_rows)

        print(f"type_{type_num} ({TYPE_NAMES[type_num]}): {len(query_rows)} queries, {len(ans_rows)} ans rows")


if __name__ == "__main__":
    generate()