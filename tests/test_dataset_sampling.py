from src.data.schema import Question
from scripts.prepare_dataset import stratified_sample


def _make_questions(n_per_type, types):
    questions = []
    i = 0
    for t in types:
        for _ in range(n_per_type):
            questions.append(
                Question(
                    question_id=f"q{i}",
                    question=f"question {i}",
                    answer="answer",
                    question_type=t,
                    domain="open",
                    dynamism="static",
                )
            )
            i += 1
    return questions


def test_sample_size_matches_request():
    questions = _make_questions(50, ["simple", "comparison", "set"])
    sampled, alloc = stratified_sample(questions, sample_size=30, seed=42)
    assert len(sampled) == 30


def test_sampling_is_reproducible_with_fixed_seed():
    questions = _make_questions(50, ["simple", "comparison", "set"])
    sampled1, _ = stratified_sample(questions, sample_size=30, seed=42)
    sampled2, _ = stratified_sample(questions, sample_size=30, seed=42)
    ids1 = [q.question_id for q in sampled1]
    ids2 = [q.question_id for q in sampled2]
    assert ids1 == ids2


def test_different_seed_can_change_sample():
    questions = _make_questions(50, ["simple", "comparison", "set"])
    sampled1, _ = stratified_sample(questions, sample_size=30, seed=42)
    sampled2, _ = stratified_sample(questions, sample_size=30, seed=7)
    ids1 = {q.question_id for q in sampled1}
    ids2 = {q.question_id for q in sampled2}
    assert ids1 != ids2


def test_proportional_allocation_roughly_even_for_equal_types():
    questions = _make_questions(50, ["simple", "comparison", "set"])
    _, alloc = stratified_sample(questions, sample_size=30, seed=42)
    assert set(alloc.keys()) == {"simple", "comparison", "set"}
    for count in alloc.values():
        assert count == 10


def test_sample_capped_by_available_questions_in_type():
    questions = _make_questions(5, ["rare_type"]) + _make_questions(95, ["common_type"])
    sampled, alloc = stratified_sample(questions, sample_size=50, seed=42)
    assert alloc["rare_type"] <= 5
    assert len(sampled) == 50
