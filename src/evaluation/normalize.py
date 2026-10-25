"""Answer normalisation shared by every correctness check.

Deterministic, rule-based (no model involved) so correctness decisions
are fully reproducible and inspectable.
"""
import re
import string

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in _ARTICLES]
    return " ".join(words)


def normalize_for_tokens(text: str) -> list:
    return normalize_answer(text).split()
