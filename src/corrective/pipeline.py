"""End-to-end per-question pipeline for both systems under evaluation:

  System A (Baseline RAG):      retrieve -> context -> generate
  System B (Corrective RAG):    retrieve -> grade evidence -> if sufficient,
                                 reuse the baseline answer; if insufficient,
                                 KG lookup -> KG evidence -> generate

Returns one flat record per question with every field required by the
evaluator and the Streamlit demo (see README "Evaluation methodology").
"""
from dataclasses import asdict, dataclass
from typing import Optional

from src.config import CFG
from src.corrective.evidence_grader import assess_evidence
from src.data.schema import Question
from src.generation.answer_generator import (
    build_kg_evidence_text,
    build_rag_evidence_text,
    generate_answer,
)
from src.knowledge_graph.kg_fallback import KGResult, kg_lookup
from src.knowledge_graph.wikidata_client import WikidataClient
from src.retrieval.tfidf_retriever import retrieve


@dataclass
class PipelineResult:
    question_id: str
    question: str
    question_type: str
    domain: str
    ground_truth: str

    retrieved_titles: list
    retrieval_scores: list
    retrieval_context: str

    baseline_answer: str

    fallback_triggered: bool
    fallback_reasons: list
    entity_coverage: float
    required_entities: list
    missing_entities: list

    kg_entities: list
    kg_qids: list
    kg_triples: list
    kg_found: bool

    corrective_answer: str


def run_pipeline(question: Question, wikidata_client: Optional[WikidataClient] = None) -> PipelineResult:
    client = wikidata_client or WikidataClient()

    # --- Baseline RAG: retrieve -> context -> generate ---
    retrieved = retrieve(question, top_k=CFG.top_k)
    rag_evidence = build_rag_evidence_text(retrieved)
    baseline_answer = generate_answer(question.question, rag_evidence)

    # --- Corrective gating: is the retrieved evidence good enough? ---
    assessment = assess_evidence(question, retrieved)

    if not assessment.fallback_triggered:
        corrective_answer = baseline_answer
        kg_result = KGResult()
    else:
        kg_result = kg_lookup(question.question, client)
        kg_evidence = build_kg_evidence_text(kg_result, fallback_context=rag_evidence)
        corrective_answer = generate_answer(question.question, kg_evidence)

    return PipelineResult(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        domain=question.domain,
        ground_truth=question.answer,
        retrieved_titles=[r.passage.title for r in retrieved],
        retrieval_scores=[round(r.score, 4) for r in retrieved],
        retrieval_context=rag_evidence,
        baseline_answer=baseline_answer,
        fallback_triggered=assessment.fallback_triggered,
        fallback_reasons=assessment.fallback_reasons,
        entity_coverage=round(assessment.entity_coverage, 4),
        required_entities=assessment.required_entities,
        missing_entities=assessment.missing_entities,
        kg_entities=[e.mention for e in kg_result.linked_entities],
        kg_qids=[e.qid for e in kg_result.linked_entities if e.qid],
        kg_triples=kg_result.triples,
        kg_found=kg_result.found,
        corrective_answer=corrective_answer,
    )
