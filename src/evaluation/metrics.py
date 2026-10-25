"""Deterministic correctness metrics.

Two metrics are computed for every prediction:
  - exact_match : normalised strings are identical.
  - token_f1    : token-level F1 between normalised prediction and gold
                  (standard SQuAD-style F1), needed because CRAG gold
                  answers and model answers legitimately differ in
                  surface form (e.g. "4" vs "four 3-point attempts per
                  game") while still agreeing on content.

`is_correct` combines them with one simple, documented rule (no LLM
judge, so this is the deterministic metric required by the brief):
a prediction is correct if it exact-matches, OR its token-F1 >= 0.5,
OR the normalised gold answer is a substring of the normalised
prediction (handles concise gold answers like "yes"/"invalid question"
embedded in a slightly longer model answer).
"""
from dataclasses import dataclass

from src.evaluation.normalize import normalize_answer, normalize_for_tokens

F1_MATCH_THRESHOLD = 0.5


@dataclass
class CorrectnessResult:
    exact_match: bool
    token_f1: float
    correct: bool


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_for_tokens(pred)
    gold_tokens = normalize_for_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(pred: str, gold: str) -> CorrectnessResult:
    em = exact_match(pred, gold)
    f1 = token_f1(pred, gold)
    norm_pred, norm_gold = normalize_answer(pred), normalize_answer(gold)
    containment = bool(norm_gold) and norm_gold in norm_pred
    correct = em or f1 >= F1_MATCH_THRESHOLD or containment
    return CorrectnessResult(exact_match=em, token_f1=f1, correct=correct)
