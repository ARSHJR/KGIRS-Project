"""Lightweight local retrieval over the search_results CRAG already supplies
for each question.

CRAG ships 5 web-search passages per question (title, snippet, page text).
Respecting that structure (rather than building an unrelated external
corpus), this retriever ranks those passages against the question using
TF-IDF cosine similarity and returns the top-k with their scores.

The retriever is intentionally simple and fully local/deterministic -
no embeddings download, no API call - so it is easy to explain in a viva
and to swap out later (e.g. for a sentence-embedding retriever) without
touching any other module.
"""
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.schema import EvidencePassage, Question


@dataclass
class RetrievedPassage:
    passage: EvidencePassage
    score: float


def _passage_text(p: EvidencePassage) -> str:
    # snippet is a short human-curated summary; text is the full page.
    # Concatenating both gives TF-IDF the best chance of matching query terms
    # without being swamped by boilerplate from the full page.
    body = p.text[:2000] if p.text else ""
    return f"{p.title}. {p.snippet} {body}".strip()


def retrieve(question: Question, top_k: int) -> List[RetrievedPassage]:
    passages = [p for p in question.passages if (p.text or p.snippet)]
    if not passages:
        return []

    docs = [_passage_text(p) for p in passages]
    corpus = [question.question] + docs

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        # vocabulary ended up empty (e.g. all-stopword passages)
        return [RetrievedPassage(p, 0.0) for p in passages[:top_k]]

    query_vec = tfidf[0:1]
    doc_vecs = tfidf[1:]
    sims = cosine_similarity(query_vec, doc_vecs)[0]

    ranked = sorted(zip(passages, sims), key=lambda x: x[1], reverse=True)
    return [RetrievedPassage(p, float(s)) for p, s in ranked[:top_k]]
