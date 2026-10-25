"""Run the full 200-question Corrective-RAG vs Baseline-RAG experiment.

For every question in the fixed evaluation subset, runs both systems via
`run_pipeline`, scores both answers against the CRAG ground truth with the
deterministic evaluator, and writes one row per question to
results/predictions/predictions.jsonl (full trace) and
results/predictions/predictions.csv (flat table for quick inspection).

Re-running this script re-evaluates the exact same 200 questions (fixed
subset file + fixed pipeline), so results are reproducible.
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, PATHS
from src.corrective.pipeline import run_pipeline
from src.data.loader import load_processed_subset
from src.evaluation.metrics import evaluate_answer
from src.knowledge_graph.wikidata_client import WikidataClient


def main():
    PATHS.ensure()
    subset_path = PATHS.processed_dir / CFG.eval_subset_filename
    questions = load_processed_subset(subset_path)
    print(f"Loaded {len(questions)} questions from {subset_path}")
    print(f"LLM generation enabled (Groq): {CFG.use_llm}")

    client = WikidataClient()
    records = []
    start = time.time()

    for i, q in enumerate(questions, 1):
        result = run_pipeline(q, client)
        base_eval = evaluate_answer(result.baseline_answer, result.ground_truth)
        corr_eval = evaluate_answer(result.corrective_answer, result.ground_truth)

        record = {
            "question_id": result.question_id,
            "question": result.question,
            "question_type": result.question_type,
            "domain": result.domain,
            "dynamism": q.dynamism,
            "ground_truth": result.ground_truth,
            "retrieved_titles": result.retrieved_titles,
            "retrieval_scores": result.retrieval_scores,
            "retrieval_context": result.retrieval_context,
            "baseline_answer": result.baseline_answer,
            "fallback_triggered": result.fallback_triggered,
            "fallback_reasons": result.fallback_reasons,
            "entity_coverage": result.entity_coverage,
            "required_entities": result.required_entities,
            "missing_entities": result.missing_entities,
            "kg_entities": result.kg_entities,
            "kg_qids": result.kg_qids,
            "kg_triples": result.kg_triples,
            "kg_found": result.kg_found,
            "corrective_answer": result.corrective_answer,
            "baseline_exact_match": base_eval.exact_match,
            "baseline_f1": round(base_eval.token_f1, 4),
            "baseline_correct": base_eval.correct,
            "corrective_exact_match": corr_eval.exact_match,
            "corrective_f1": round(corr_eval.token_f1, 4),
            "corrective_correct": corr_eval.correct,
        }
        records.append(record)

        if i % 10 == 0 or i == len(questions):
            elapsed = time.time() - start
            n_fallback = sum(1 for r in records if r["fallback_triggered"])
            print(f"  [{i}/{len(questions)}] elapsed={elapsed:.1f}s fallback_so_far={n_fallback}")

    out_jsonl = PATHS.predictions_dir / "predictions.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} full prediction records to {out_jsonl}")

    out_csv = PATHS.predictions_dir / "predictions.csv"
    flat_fields = [
        "question_id", "question", "question_type", "domain", "ground_truth",
        "baseline_answer", "corrective_answer",
        "baseline_correct", "corrective_correct",
        "baseline_exact_match", "corrective_exact_match",
        "baseline_f1", "corrective_f1",
        "fallback_triggered", "fallback_reasons",
        "entity_coverage", "kg_found", "kg_entities",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["fallback_reasons"] = ";".join(row["fallback_reasons"])
            row["kg_entities"] = ";".join(row["kg_entities"])
            writer.writerow(row)
    print(f"Saved flat CSV to {out_csv}")


if __name__ == "__main__":
    main()
