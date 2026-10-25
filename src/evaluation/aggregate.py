"""Aggregate per-question predictions into the full metrics suite required
by the project brief: overall accuracy, fallback rate/effectiveness, and
per-question-type breakdowns.

Takes a list of per-question dicts (each must have the fields produced by
`run_evaluation.py` / `PipelineResult` plus correctness flags) and returns
plain-dict summaries that are trivial to dump to CSV/JSON and to plot.
"""
from collections import defaultdict
from typing import List


def _safe_div(a, b):
    return a / b if b else 0.0


def _accuracy(records, key):
    if not records:
        return 0.0
    return sum(1 for r in records if r[key]) / len(records)


def overall_metrics(records: List[dict]) -> dict:
    n = len(records)
    baseline_correct = [r for r in records if r["baseline_correct"]]
    corrective_correct = [r for r in records if r["corrective_correct"]]
    fallback_recs = [r for r in records if r["fallback_triggered"]]
    no_fallback_recs = [r for r in records if not r["fallback_triggered"]]

    baseline_acc = _accuracy(records, "baseline_correct")
    corrective_acc = _accuracy(records, "corrective_correct")

    improved = [r for r in records if not r["baseline_correct"] and r["corrective_correct"]]
    worsened = [r for r in records if r["baseline_correct"] and not r["corrective_correct"]]
    unchanged = [r for r in records if r["baseline_correct"] == r["corrective_correct"]]

    return {
        "num_questions": n,
        "baseline_accuracy": round(baseline_acc, 4),
        "corrective_accuracy": round(corrective_acc, 4),
        "absolute_improvement": round(corrective_acc - baseline_acc, 4),
        "relative_improvement_pct": round(
            _safe_div(corrective_acc - baseline_acc, baseline_acc) * 100, 2
        ) if baseline_acc > 0 else None,
        "total_kg_fallback_count": len(fallback_recs),
        "kg_fallback_rate": round(_safe_div(len(fallback_recs), n), 4),
        "accuracy_when_fallback_triggered": round(_accuracy(fallback_recs, "corrective_correct"), 4),
        "accuracy_when_fallback_not_triggered": round(_accuracy(no_fallback_recs, "corrective_correct"), 4),
        "fallback_success_rate": round(
            _safe_div(sum(1 for r in fallback_recs if r["corrective_correct"]), len(fallback_recs)), 4
        ),
        "num_baseline_wrong_corrective_correct": len(improved),
        "num_baseline_correct_corrective_wrong": len(worsened),
        "num_unchanged": len(unchanged),
    }


def confusion_breakdown(records: List[dict]) -> dict:
    bc_cc = sum(1 for r in records if r["baseline_correct"] and r["corrective_correct"])
    bw_cc = sum(1 for r in records if not r["baseline_correct"] and r["corrective_correct"])
    bc_cw = sum(1 for r in records if r["baseline_correct"] and not r["corrective_correct"])
    bw_cw = sum(1 for r in records if not r["baseline_correct"] and not r["corrective_correct"])
    return {
        "baseline_correct_corrective_correct": bc_cc,
        "baseline_wrong_corrective_correct": bw_cc,
        "baseline_correct_corrective_wrong": bc_cw,
        "baseline_wrong_corrective_wrong": bw_cw,
    }


def per_question_type_metrics(records: List[dict]) -> List[dict]:
    by_type = defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r)

    rows = []
    for qtype, recs in sorted(by_type.items()):
        n = len(recs)
        baseline_acc = _accuracy(recs, "baseline_correct")
        corrective_acc = _accuracy(recs, "corrective_correct")
        fallback_recs = [r for r in recs if r["fallback_triggered"]]
        rows.append({
            "question_type": qtype,
            "num_questions": n,
            "baseline_accuracy": round(baseline_acc, 4),
            "corrective_accuracy": round(corrective_acc, 4),
            "kg_fallback_count": len(fallback_recs),
            "kg_fallback_rate": round(_safe_div(len(fallback_recs), n), 4),
            "accuracy_gain": round(corrective_acc - baseline_acc, 4),
        })
    return rows


def per_dynamism_metrics(records: List[dict]) -> List[dict]:
    """Breaks results down by CRAG's static/slow-changing/fast-changing/
    real-time label. Useful because the Knowledge-Graph fallback (a
    largely static snapshot) is structurally unable to help with
    fast-changing or real-time facts (e.g. live stock prices) - this
    table makes that limitation visible in the numbers rather than
    hiding it."""
    by_dyn = defaultdict(list)
    for r in records:
        by_dyn[r.get("dynamism", "unknown")].append(r)

    rows = []
    for dyn, recs in sorted(by_dyn.items()):
        n = len(recs)
        fallback_recs = [r for r in recs if r["fallback_triggered"]]
        rows.append({
            "dynamism": dyn,
            "num_questions": n,
            "baseline_accuracy": round(_accuracy(recs, "baseline_correct"), 4),
            "corrective_accuracy": round(_accuracy(recs, "corrective_correct"), 4),
            "kg_fallback_rate": round(_safe_div(len(fallback_recs), n), 4),
            "accuracy_when_fallback_triggered": round(_accuracy(fallback_recs, "corrective_correct"), 4),
        })
    return rows


def per_domain_metrics(records: List[dict]) -> List[dict]:
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    rows = []
    for domain, recs in sorted(by_domain.items()):
        n = len(recs)
        rows.append({
            "domain": domain,
            "num_questions": n,
            "baseline_accuracy": round(_accuracy(recs, "baseline_correct"), 4),
            "corrective_accuracy": round(_accuracy(recs, "corrective_correct"), 4),
            "kg_fallback_rate": round(
                _safe_div(sum(1 for r in recs if r["fallback_triggered"]), n), 4
            ),
        })
    return rows
