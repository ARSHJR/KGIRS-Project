"""Build the fixed 200-question CRAG evaluation subset.

Samples are drawn stratified by `question_type` (CRAG's official taxonomy:
simple, simple_w_condition, multi-hop, post-processing, false_premise,
comparison, set, aggregation) proportionally to each type's share of the
combined raw pool, using a fixed random seed so repeated runs select the
exact same 200 questions.
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, PATHS
from src.data.loader import load_raw_crag_dir, save_processed_subset


def stratified_sample(questions, sample_size: int, seed: int):
    rng = random.Random(seed)
    by_type = {}
    for q in questions:
        by_type.setdefault(q.question_type, []).append(q)
    for v in by_type.values():
        rng.shuffle(v)

    total = len(questions)
    types_sorted = sorted(by_type.keys())

    # proportional allocation with largest-remainder method so counts sum
    # exactly to sample_size while staying proportional to each type's share
    raw_alloc = {t: sample_size * len(by_type[t]) / total for t in types_sorted}
    base_alloc = {t: int(raw_alloc[t]) for t in types_sorted}
    remainder = sample_size - sum(base_alloc.values())
    remainders_sorted = sorted(types_sorted, key=lambda t: raw_alloc[t] - base_alloc[t], reverse=True)
    for t in remainders_sorted[:remainder]:
        base_alloc[t] += 1

    # cap allocation to what's available per type, redistribute shortfall
    final_alloc = {}
    shortfall = 0
    for t in types_sorted:
        want = base_alloc[t]
        have = len(by_type[t])
        final_alloc[t] = min(want, have)
        shortfall += max(0, want - have)
    if shortfall > 0:
        donors = [t for t in types_sorted if len(by_type[t]) > final_alloc[t]]
        rng.shuffle(donors)
        i = 0
        while shortfall > 0 and donors:
            t = donors[i % len(donors)]
            if len(by_type[t]) > final_alloc[t]:
                final_alloc[t] += 1
                shortfall -= 1
            i += 1
            if i > 10000:
                break

    sampled = []
    for t in types_sorted:
        sampled.extend(by_type[t][: final_alloc[t]])
    rng.shuffle(sampled)
    return sampled, final_alloc


def main():
    PATHS.ensure()
    print(f"Loading raw CRAG files: {CFG.raw_subset_files}")
    questions = load_raw_crag_dir(PATHS.raw_dir, list(CFG.raw_subset_files))
    print(f"Loaded {len(questions)} raw questions.")
    print("Question type distribution (raw pool):", Counter(q.question_type for q in questions))
    print("Domain distribution (raw pool):", Counter(q.domain for q in questions))

    sampled, alloc = stratified_sample(questions, CFG.sample_size, CFG.random_seed)
    print(f"\nSampled {len(sampled)} questions with seed={CFG.random_seed}")
    print("Per-type allocation:", alloc)
    print("Sampled domain distribution:", Counter(q.domain for q in sampled))
    print("Sampled dynamism distribution:", Counter(q.dynamism for q in sampled))

    out_path = PATHS.processed_dir / CFG.eval_subset_filename
    save_processed_subset(sampled, out_path)
    print(f"\nSaved fixed evaluation subset to {out_path}")


if __name__ == "__main__":
    main()
