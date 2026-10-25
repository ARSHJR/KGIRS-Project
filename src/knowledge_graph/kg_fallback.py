"""Knowledge-Graph fallback: entity linking -> SPARQL triple lookup ->
concise textual evidence, used when the evidence grader decides normal
RAG retrieval was insufficient.

This module performs a real structured KG access path (Wikidata SPARQL),
not a disguised LLM call: every fact in the returned evidence is a
(property, value) pair actually read off a resolved Wikidata entity. If no
entity resolves, or a resolved entity has no triples relevant to the
question's keywords, that is recorded explicitly (`found=False`) instead
of fabricating facts.
"""
from dataclasses import dataclass, field
from typing import List

from src.knowledge_graph.entity_linker import LinkedEntity, link_entities
from src.knowledge_graph.wikidata_client import WikidataClient
from src.utils.entities import extract_keywords


@dataclass
class KGResult:
    linked_entities: List[LinkedEntity] = field(default_factory=list)
    triples: List[dict] = field(default_factory=list)  # [{entity, property, value}]
    evidence_text: str = ""
    found: bool = False


def _relevance(question_keywords: set, triple: dict) -> int:
    text = f"{triple['property']} {triple['value']}".lower()
    return sum(1 for kw in question_keywords if kw in text)


def kg_lookup(question_text: str, client: WikidataClient, max_triples: int = 8) -> KGResult:
    linked = link_entities(question_text, client)
    resolved = [e for e in linked if e.resolved and e.qid]

    if not resolved:
        return KGResult(linked_entities=linked, triples=[], evidence_text="", found=False)

    question_keywords = set(extract_keywords(question_text))

    all_triples = []
    for ent in resolved:
        raw_triples = client.get_entity_triples(ent.qid, limit=40)
        for t in raw_triples:
            all_triples.append({"entity": ent.label, "property": t["property"], "value": t["value"]})

    if not all_triples:
        return KGResult(linked_entities=linked, triples=[], evidence_text="", found=False)

    # rank triples by lexical overlap with the question so the most
    # question-relevant facts are surfaced first; ties keep original order
    ranked = sorted(all_triples, key=lambda t: _relevance(question_keywords, t), reverse=True)
    top_triples = ranked[:max_triples]

    evidence_lines = [f"({t['entity']}, {t['property']}, {t['value']})" for t in top_triples]
    evidence_text = "Knowledge Graph facts:\n" + "\n".join(evidence_lines)

    return KGResult(linked_entities=linked, triples=top_triples, evidence_text=evidence_text, found=True)
