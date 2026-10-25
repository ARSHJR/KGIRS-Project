from src.evaluation.normalize import normalize_answer


def test_lowercase_and_strip():
    assert normalize_answer("  Paris  ") == "paris"


def test_removes_punctuation():
    assert normalize_answer("Steve Nash, Jr.") == "steve nash jr"


def test_removes_articles():
    assert normalize_answer("The Dark Knight") == "dark knight"
    assert normalize_answer("a lion") == "lion"
    assert normalize_answer("an apple") == "apple"


def test_none_and_empty():
    assert normalize_answer(None) == ""
    assert normalize_answer("") == ""


def test_whitespace_collapsing():
    assert normalize_answer("New   York    City") == "new york city"
