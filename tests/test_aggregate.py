from src.evaluation.aggregate import (
    confusion_breakdown,
    overall_metrics,
    per_question_type_metrics,
)


def _rec(qtype, domain, baseline_correct, corrective_correct, fallback):
    return {
        "question_type": qtype,
        "domain": domain,
        "baseline_correct": baseline_correct,
        "corrective_correct": corrective_correct,
        "fallback_triggered": fallback,
    }


RECORDS = [
    _rec("simple", "open", True, True, False),
    _rec("simple", "open", False, True, True),  # improved by fallback
    _rec("simple", "open", True, False, True),  # worsened by fallback
    _rec("comparison", "movie", False, False, True),
    _rec("comparison", "movie", False, False, False),
]


def test_overall_metrics_counts():
    m = overall_metrics(RECORDS)
    assert m["num_questions"] == 5
    assert m["baseline_accuracy"] == 2 / 5
    assert m["corrective_accuracy"] == 2 / 5
    assert m["total_kg_fallback_count"] == 3
    assert m["kg_fallback_rate"] == 3 / 5
    assert m["num_baseline_wrong_corrective_correct"] == 1
    assert m["num_baseline_correct_corrective_wrong"] == 1


def test_confusion_breakdown_sums_to_total():
    c = confusion_breakdown(RECORDS)
    assert sum(c.values()) == len(RECORDS)
    assert c["baseline_wrong_corrective_correct"] == 1
    assert c["baseline_correct_corrective_wrong"] == 1


def test_per_question_type_metrics_groups_correctly():
    rows = per_question_type_metrics(RECORDS)
    by_type = {r["question_type"]: r for r in rows}
    assert by_type["simple"]["num_questions"] == 3
    assert by_type["comparison"]["num_questions"] == 2
    assert by_type["simple"]["kg_fallback_count"] == 2
