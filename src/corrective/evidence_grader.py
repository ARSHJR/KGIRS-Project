"""Evidence-quality grading: the explicit, explainable decision of whether
retrieved RAG evidence is good enough to answer from, or whether Corrective
RAG should fall back to a Knowledge-Graph lookup.

Design (deliberately simple and explainable - see README "Corrective/
fallback decision logic" for the rationale):

  1. retrieval_score   - top-1 TF-IDF cosine similarity between the
                          question and the best retrieved passage.
  2. entity_coverage    - fraction of the question's key entities/keywords
                          that actually appear (literal substring match) in
                          the retrieved passages' text. This catches the
                          common failure mode where a passage is lexically
                          similar (shares generic words) but never actually
                          mentions the entity the question is about.

Fallback triggers (any one is sufficient; all reasons that fire are kept,
not just the first, so the trace is complete):
  - "no_evidence_retrieved"     : zero passages came back from retrieval.
  - "low_retrieval_relevance"   : retrieval_score below threshold.
  - "missing_required_entity"   : entity_coverage below threshold.

No LLM call is used for this decision - it must be deterministic and
reproducible, and inspectable for a viva without pointing at an opaque
model call.
"""
from dataclasses import dataclass, field
from typing import List

from src.config import CFG
from src.data.schema import Question
from src.retrieval.tfidf_retriever import RetrievedPassage
from src.utils.entities import extract_entities, extract_keywords


@dataclass
class EvidenceAssessment:
    retrieval_score: float
    entity_coverage: float
    required_entities: List[str]
    covered_entities: List[str]
    missing_entities: List[str]
    fallback_triggered: bool
    fallback_reasons: List[str] = field(default_factory=list)


def _entity_mentioned(entity: str, passages_lower: str) -> bool:
    """An entity counts as mentioned if the full phrase appears verbatim,
    or - for multi-word entities - at least half of its significant words
    (len>2) appear in the passage text. The partial-match path matters
    because a passage can clearly be about an entity ("Titanic is a 1997
    film directed by...") without echoing a generic head noun the
    question used ("the movie Titanic")."""
    entity_lower = entity.lower()
    if entity_lower in passages_lower:
        return True
    words = [w for w in entity_lower.split() if len(w) > 2]
    if not words:
        return False
    hits = sum(1 for w in words if w in passages_lower)
    return hits / len(words) >= 0.5


def _entity_coverage(question_entities: List[str], passages_text: str):
    passages_lower = passages_text.lower()
    covered, missing = [], []
    for ent in question_entities:
        if _entity_mentioned(ent, passages_lower):
            covered.append(ent)
        else:
            missing.append(ent)
    coverage = len(covered) / len(question_entities) if question_entities else 1.0
    return coverage, covered, missing


def assess_evidence(question: Question, retrieved: List[RetrievedPassage]) -> EvidenceAssessment:
    if not retrieved:
        entities = extract_entities(question.question) or extract_keywords(question.question)
        return EvidenceAssessment(
            retrieval_score=0.0,
            entity_coverage=0.0,
            required_entities=entities,
            covered_entities=[],
            missing_entities=entities,
            fallback_triggered=True,
            fallback_reasons=["no_evidence_retrieved"],
        )

    retrieval_score = retrieved[0].score

    entities = extract_entities(question.question)
    if not entities:
        entities = extract_keywords(question.question)[:5]

    passages_text = " ".join(
        f"{r.passage.title} {r.passage.snippet} {r.passage.text}" for r in retrieved
    )
    coverage, covered, missing = _entity_coverage(entities, passages_text)

    reasons = []
    if retrieval_score < CFG.fallback_similarity_threshold:
        reasons.append("low_retrieval_relevance")
    if coverage < CFG.fallback_entity_coverage_threshold:
        reasons.append("missing_required_entity")

    return EvidenceAssessment(
        retrieval_score=retrieval_score,
        entity_coverage=coverage,
        required_entities=entities,
        covered_entities=covered,
        missing_entities=missing,
        fallback_triggered=len(reasons) > 0,
        fallback_reasons=reasons,
    )
