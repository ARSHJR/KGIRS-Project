"""Answer generation, modular over two backends selected by CFG.use_llm:

  - Groq LLM (when GROQ_API_KEY is set): prompted strictly to answer only
    from the supplied evidence, concisely, and to say "I don't know" if
    the evidence does not support an answer - this is what keeps the
    comparison honest: the generator never sees the ground-truth answer,
    only retrieved/KG evidence.
  - Local extractive fallback (no API key needed, used by default): picks
    the single sentence from the evidence most lexically similar to the
    question (TF-IDF cosine) as the answer. Fully deterministic/offline.

Both paths are used identically by Baseline RAG (context-only) and by
Corrective RAG's KG-fallback path (KG facts, optionally plus context).
"""
import re
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CFG
from src.generation.llm_client import LLMUnavailableError, chat_complete

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. Answer the question "
    "using ONLY the evidence provided below. Do not use outside knowledge. "
    "If the evidence does not contain the answer, reply exactly: I don't know. "
    "Be extremely concise: answer in as few words as possible (a name, number, "
    "date, or short phrase) with no explanation, no restating the question."
)


_TITLE_TAG_RE = re.compile(r"^\[[^\]]*\]\s*")
_HEADER_LINE_RE = re.compile(r"^(Knowledge Graph facts:|Additional web context:)\s*$")


def _split_sentences(text: str) -> List[str]:
    """Split evidence text into fine-grained candidate answer units: each
    line (a KG triple, or a passage line) is its own unit, further split
    on sentence punctuation. Treating KG triple lines as whole units
    (rather than joining them and only then splitting on '.') keeps a
    single "(entity, property, value)" fact from being truncated mid-tuple
    by the final length cap."""
    units = []
    for line in text.split("\n"):
        line = _TITLE_TAG_RE.sub("", line).strip()
        if not line or _HEADER_LINE_RE.match(line):
            continue
        for s in _SENTENCE_SPLIT_RE.split(line):
            s = s.strip()
            if len(s) > 3:
                units.append(s)
    return units


def _extractive_answer(question: str, evidence_text: str) -> str:
    """No-API fallback: return the sentence in the evidence most similar
    to the question, trimmed to a reasonable length."""
    sentences = _split_sentences(evidence_text)
    if not sentences:
        return "I don't know"

    corpus = [question] + sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        return sentences[0][:200]

    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    best_idx = int(sims.argmax())
    if sims[best_idx] <= 0:
        return "I don't know"
    return sentences[best_idx][:200]


def _llm_answer(question: str, evidence_text: str) -> str:
    user_prompt = f"Evidence:\n{evidence_text}\n\nQuestion: {question}\nAnswer:"
    try:
        return chat_complete(_SYSTEM_PROMPT, user_prompt)
    except (LLMUnavailableError, Exception):
        return _extractive_answer(question, evidence_text)


def generate_answer(question: str, evidence_text: str) -> str:
    if not evidence_text.strip():
        return "I don't know"
    if CFG.use_llm:
        return _llm_answer(question, evidence_text)
    return _extractive_answer(question, evidence_text)


def build_rag_evidence_text(retrieved_passages) -> str:
    parts = []
    for r in retrieved_passages:
        snippet = r.passage.snippet or r.passage.text[:500]
        parts.append(f"[{r.passage.title}] {snippet}")
    return "\n".join(parts)


def build_kg_evidence_text(kg_result, fallback_context: Optional[str] = None) -> str:
    parts = []
    if kg_result.found:
        parts.append(kg_result.evidence_text)
    if fallback_context:
        parts.append("Additional web context:\n" + fallback_context)
    return "\n\n".join(parts)
