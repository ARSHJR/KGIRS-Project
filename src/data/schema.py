"""Normalised dataset schema.

The CRAG benchmark's raw records are converted into this stable shape so
that every downstream module (retrieval, corrective logic, evaluation)
depends on one schema instead of the raw CRAG field names. If a future
CRAG release changes field names, only `loader.py` needs to change.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidencePassage:
    title: str
    url: str
    snippet: str
    text: str  # full page_result text (HTML-stripped) if available


@dataclass
class Question:
    question_id: str
    question: str
    answer: str
    question_type: str  # official CRAG category: simple, simple_w_condition,
    # multi-hop, post-processing, false_premise, comparison, set, aggregation
    domain: str  # finance, movie, music, sports, open
    dynamism: str  # static, slow-changing, fast-changing, real-time
    query_time: Optional[str] = None
    passages: List[EvidencePassage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "domain": self.domain,
            "dynamism": self.dynamism,
            "query_time": self.query_time,
            "passages": [vars(p) for p in self.passages],
        }

    @staticmethod
    def from_dict(d: dict) -> "Question":
        passages = [EvidencePassage(**p) for p in d.get("passages", [])]
        return Question(
            question_id=d["question_id"],
            question=d["question"],
            answer=d["answer"],
            question_type=d["question_type"],
            domain=d["domain"],
            dynamism=d["dynamism"],
            query_time=d.get("query_time"),
            passages=passages,
        )
