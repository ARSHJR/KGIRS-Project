from unittest.mock import MagicMock

from src.knowledge_graph.entity_linker import link_entities
from src.knowledge_graph.kg_fallback import kg_lookup


def test_link_entities_no_candidates_marks_unresolved():
    client = MagicMock()
    client.search_entity.return_value = []
    linked = link_entities("who founded the company Acme Corp", client)
    assert all(not e.resolved for e in linked)


def test_link_entities_resolves_top_candidate():
    client = MagicMock()
    client.search_entity.return_value = [
        {"id": "Q1", "label": "Acme Corp", "description": "a company"}
    ]
    linked = link_entities("who founded Acme Corp", client)
    resolved = [e for e in linked if e.resolved]
    assert len(resolved) >= 1
    assert resolved[0].qid == "Q1"


def test_kg_lookup_no_entities_resolved_returns_not_found():
    client = MagicMock()
    client.search_entity.return_value = []
    result = kg_lookup("asdkj qwoeiu zxcvb", client)
    assert not result.found
    assert result.triples == []


def test_kg_lookup_builds_evidence_from_triples():
    client = MagicMock()
    client.search_entity.return_value = [
        {"id": "Q312", "label": "Apple Inc.", "description": "technology company"}
    ]
    client.get_entity_triples.return_value = [
        {"property": "founder", "value": "Steve Jobs"},
        {"property": "industry", "value": "Technology"},
    ]
    result = kg_lookup("who founded Apple Inc.", client)
    assert result.found
    assert any(t["property"] == "founder" for t in result.triples)
    assert "Steve Jobs" in result.evidence_text
