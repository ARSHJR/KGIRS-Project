"""Shared entity-extraction utility.

Used by (a) the corrective evidence grader, to check whether retrieved
passages mention the entities the question is actually about, and (b) the
Knowledge-Graph entity-linking stage, to pick candidate entity strings to
resolve against Wikidata.

Uses spaCy's small English model for named-entity recognition, with a
noun-chunk fallback for questions where spaCy finds no named entities
(common for CRAG's "simple_w_condition"/"aggregation" style questions that
refer to common nouns like "the lead actor" rather than proper nouns).
"""
import re
from functools import lru_cache
from typing import List

import spacy

_STOPWORD_LIKE = {"the", "a", "an", "this", "that", "these", "those"}

# CRAG questions are almost entirely lowercase ("who directed titanic"),
# but spaCy's statistical NER relies heavily on capitalization cues and
# performs poorly on all-lowercase input. A lightweight truecasing pass -
# capitalizing every word that is not a common function word - restores
# enough of that signal for NER to correctly recognize proper nouns
# (e.g. "steve nash" -> "Steve Nash" -> PERSON) without any per-question
# hardcoding.
_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "how", "many", "much", "what", "who", "whom", "when", "where", "which",
    "why", "and", "or", "but", "that", "this", "these", "those", "of",
    "in", "on", "at", "to", "for", "per", "he", "she", "it", "they", "him",
    "her", "them", "his", "its", "their", "made", "make", "with", "by",
    "from", "as", "be", "been", "has", "have", "had", "than", "then",
    "average", "total", "number", "list", "name", "all",
}


def _truecase(text: str) -> str:
    words = text.split()
    out = []
    for w in words:
        bare = w.strip("?.,!'\"")
        if not bare or bare.lower() in _FUNCTION_WORDS:
            out.append(w)
        else:
            out.append(w[0].upper() + w[1:] if w else w)
    return " ".join(out)


@lru_cache(maxsize=1)
def _get_nlp():
    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


# numeric/quantity entity types carry no useful identity for KG linking
_SKIP_ENT_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "TIME", "DATE"}
# entity types most likely to resolve to a meaningful Wikidata item
_PRIORITY_ENT_LABELS = {"PERSON": 0, "ORG": 1, "GPE": 1, "NORP": 1, "FAC": 1,
                         "LOC": 1, "WORK_OF_ART": 1, "EVENT": 1, "LAW": 2, "PRODUCT": 2}


def extract_entities(text: str, max_entities: int = 5) -> List[str]:
    """Return candidate entity surface forms, named entities first (ranked
    by entity-type priority), then noun chunks, de-duplicated and capped
    at max_entities."""
    nlp = _get_nlp()
    doc = nlp(_truecase(text))

    ents = [e for e in doc.ents if e.label_ not in _SKIP_ENT_LABELS]
    ents.sort(key=lambda e: _PRIORITY_ENT_LABELS.get(e.label_, 3))

    candidates = []
    seen = set()

    for ent in ents:
        key = ent.text.lower().strip()
        if key and key not in seen and key not in _STOPWORD_LIKE and key not in _FUNCTION_WORDS:
            candidates.append(ent.text.strip())
            seen.add(key)

    for chunk in doc.noun_chunks:
        words = [w for w in chunk.text.split() if w.lower() not in _STOPWORD_LIKE and w.lower() not in _FUNCTION_WORDS]
        cleaned = " ".join(words).strip()
        key = cleaned.lower()
        if cleaned and key not in seen and len(cleaned) > 2:
            candidates.append(cleaned)
            seen.add(key)

    return candidates[:max_entities]


def extract_keywords(text: str) -> List[str]:
    """Simple content-word extraction (lowercased, alnum, length>2, not a
    stopword-like filler) - used for lexical entity-coverage scoring."""
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORD_LIKE]
