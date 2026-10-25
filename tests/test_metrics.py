from src.evaluation.metrics import evaluate_answer, exact_match, token_f1


def test_exact_match_true():
    assert exact_match("Paris", "paris")


def test_exact_match_false():
    assert not exact_match("Paris", "London")


def test_token_f1_partial_overlap():
    f1 = token_f1("Steve Nash scored 4 points", "4 points")
    assert 0.0 < f1 < 1.0


def test_token_f1_identical():
    assert token_f1("invalid question", "invalid question") == 1.0


def test_token_f1_no_overlap():
    assert token_f1("apple", "banana") == 0.0


def test_token_f1_empty_both():
    assert token_f1("", "") == 1.0


def test_token_f1_one_empty():
    assert token_f1("", "paris") == 0.0


def test_evaluate_answer_exact():
    result = evaluate_answer("Paris", "Paris")
    assert result.correct
    assert result.exact_match


def test_evaluate_answer_containment():
    result = evaluate_answer("The answer is invalid question.", "invalid question")
    assert result.correct
    assert not result.exact_match


def test_evaluate_answer_wrong():
    result = evaluate_answer("Berlin", "Paris")
    assert not result.correct
