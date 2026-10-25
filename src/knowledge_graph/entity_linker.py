"""Entity linking: question text -> candidate Wikidata QIDs.

Not research-grade (no learned disambiguation model) but technically
meaningful: it extracts candidate entity mentions with spaCy NER/noun
chunks (see src/utils/entities.py), then resolves each mention to a
Wikidata QID via the wbsearchentities API, keeping the top-ranked
candidate returned by Wikidata's own relevance ranking. Nothing is
hardcoded per-question - the same code path runs for any input text.
"""
from dataclasses import dataclass
from typing import List, Optional

from src.config import CFG
from src.knowledge_graph.wikidata_client import WikidataClient
from src.utils.entities import extract_entities, extract_keywords


@dataclass
class LinkedEntity:
    mention: str
    qid: Optional[str]
    label: Optional[str]
    description: Optional[str]
    resolved: bool


def link_entities(question_text: str, client: WikidataClient) -> List[LinkedEntity]:
    mentions = extract_entities(question_text, max_entities=CFG.kg_max_candidate_entities)
    if not mentions:
        mentions = extract_keywords(question_text)[: CFG.kg_max_candidate_entities]

    question_keywords = set(extract_keywords(question_text))

    linked = []
    for mention in mentions:
        candidates = client.search_entity(mention, limit=5)
        if candidates:
            top = _best_candidate(candidates, question_keywords)
            linked.append(
                LinkedEntity(
                    mention=mention,
                    qid=top["id"],
                    label=top["label"],
                    description=top.get("description", ""),
                    resolved=True,
                )
            )
        else:
            linked.append(
                LinkedEntity(mention=mention, qid=None, label=None, description=None, resolved=False)
            )
    return linked


def _best_candidate(candidates: List[dict], question_keywords: set) -> dict:
    """Disambiguate same-name entities (e.g. multiple people named "Steve
    Nash") by preferring the Wikidata candidate whose description shares
    the most keywords with the question, falling back to Wikidata's own
    top-ranked result (candidates[0]) when no candidate's description
    overlaps with the question at all."""
    best, best_score = candidates[0], -1
    for cand in candidates:
        desc_keywords = set(extract_keywords(cand.get("description", "")))
        score = len(desc_keywords & question_keywords)
        if score > best_score:
            best, best_score = cand, score
    return best if best_score > 0 else candidates[0]
