"""Manual smoke test: run the full pipeline on a handful of questions and
print a trace of every decision, to sanity-check behaviour before the full
200-question run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, PATHS
from src.corrective.pipeline import run_pipeline
from src.data.loader import load_processed_subset
from src.evaluation.metrics import evaluate_answer
from src.knowledge_graph.wikidata_client import WikidataClient


def main(n=6):
    questions = load_processed_subset(PATHS.processed_dir / CFG.eval_subset_filename)
    client = WikidataClient()

    for q in questions[:n]:
        r = run_pipeline(q, client)
        base_eval = evaluate_answer(r.baseline_answer, r.ground_truth)
        corr_eval = evaluate_answer(r.corrective_answer, r.ground_truth)

        print("=" * 90)
        print(f"[{r.question_type} | {r.domain}] {r.question}")
        print(f"  Gold: {r.ground_truth!r}")
        print(f"  Retrieval scores: {r.retrieval_scores}")
        print(f"  Baseline answer: {r.baseline_answer!r}  correct={base_eval.correct}")
        print(f"  Fallback triggered: {r.fallback_triggered} reasons={r.fallback_reasons}")
        print(f"  Entity coverage: {r.entity_coverage} missing={r.missing_entities}")
        if r.fallback_triggered:
            print(f"  KG entities: {r.kg_entities} qids={r.kg_qids} found={r.kg_found}")
            print(f"  KG triples: {r.kg_triples[:5]}")
        print(f"  Corrective answer: {r.corrective_answer!r}  correct={corr_eval.correct}")


if __name__ == "__main__":
    main()
