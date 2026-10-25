"""Dataset-loading abstraction.

Isolates the rest of the pipeline from the raw CRAG file format. The raw
Quivr/CRAG mirror stores each subset as a bz2 file containing pretty-printed
JSON objects concatenated back-to-back (not strict single-line JSONL), so a
custom streaming parser is needed instead of plain `json.loads` per line.

If CRAG's schema changes in a future release, only `load_raw_crag_file`
and `_record_to_question` need to change — everything downstream consumes
the stable `Question` dataclass from schema.py.
"""
import bz2
import json
from pathlib import Path
from typing import Iterator, List

from src.data.html_utils import html_to_text
from src.data.schema import EvidencePassage, Question


def _iter_json_objects(text: str) -> Iterator[dict]:
    """Yield each JSON object from a string of concatenated pretty-printed
    JSON values (no separators, no enclosing array)."""
    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \n\r\t":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        yield obj
        idx = end


def load_raw_crag_file(path: Path, strip_html: bool = True) -> List[Question]:
    """Load one bz2-compressed CRAG subset file into Question objects."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        raw_text = f.read()

    questions = []
    for rec in _iter_json_objects(raw_text):
        questions.append(_record_to_question(rec, strip_html=strip_html))
    return questions


def _record_to_question(rec: dict, strip_html: bool = True) -> Question:
    passages = []
    for sr in rec.get("search_results", []) or []:
        text = sr.get("page_result", "") or ""
        if strip_html and text:
            text = html_to_text(text)
        passages.append(
            EvidencePassage(
                title=sr.get("page_name", "") or "",
                url=sr.get("page_url", "") or "",
                snippet=sr.get("page_snippet", "") or "",
                text=text,
            )
        )
    return Question(
        question_id=rec.get("interaction_id", ""),
        question=rec.get("query", "").strip(),
        answer=(rec.get("answer", "") or "").strip(),
        question_type=rec.get("question_type", "unknown"),
        domain=rec.get("domain", "unknown"),
        dynamism=rec.get("static_or_dynamic", "unknown"),
        query_time=rec.get("query_time"),
        passages=passages,
    )


def load_raw_crag_dir(raw_dir: Path, filenames: List[str]) -> List[Question]:
    all_questions: List[Question] = []
    for fname in filenames:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected raw CRAG file not found: {fpath}. "
                "Download the Quivr/CRAG dataset subset files first."
            )
        all_questions.extend(load_raw_crag_file(fpath))
    return all_questions


def load_processed_subset(path: Path) -> List[Question]:
    """Load the fixed 200-question evaluation subset saved as JSONL."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(Question.from_dict(json.loads(line)))
    return questions


def save_processed_subset(questions: List[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
