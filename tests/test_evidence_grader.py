from src.corrective.evidence_grader import assess_evidence
from src.data.schema import EvidencePassage, Question
from src.retrieval.tfidf_retriever import RetrievedPassage


def _question(text="who directed the movie Titanic?"):
    return Question(
        question_id="q1",
        question=text,
        answer="James Cameron",
        question_type="simple",
        domain="movie",
        dynamism="static",
    )


def test_no_evidence_triggers_fallback():
    assessment = assess_evidence(_question(), [])
    assert assessment.fallback_triggered
    assert "no_evidence_retrieved" in assessment.fallback_reasons


def test_relevant_evidence_does_not_trigger_fallback():
    passage = EvidencePassage(
        title="Titanic (1997 film)",
        url="http://example.com",
        snippet="Titanic is a 1997 film directed by James Cameron.",
        text="Titanic is a 1997 American epic romance film directed by James Cameron, starring Leonardo DiCaprio.",
    )
    retrieved = [RetrievedPassage(passage=passage, score=0.9)]
    assessment = assess_evidence(_question(), retrieved)
    assert not assessment.fallback_triggered
    assert assessment.entity_coverage == 1.0


def test_irrelevant_evidence_triggers_fallback():
    passage = EvidencePassage(
        title="Unrelated page",
        url="http://example.com",
        snippet="This page is about cooking recipes.",
        text="A great recipe for pasta with tomato sauce and basil.",
    )
    retrieved = [RetrievedPassage(passage=passage, score=0.02)]
    assessment = assess_evidence(_question(), retrieved)
    assert assessment.fallback_triggered
    assert "low_retrieval_relevance" in assessment.fallback_reasons
