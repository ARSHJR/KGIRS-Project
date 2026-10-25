"""Streamlit demo for the Corrective RAG with KG fallback project.

Two tabs:
  - "Try a question": runs the live pipeline on any question the user
    types, showing every intermediate step (retrieval, evidence grading,
    fallback decision, KG entity linking, KG facts, both final answers).
  - "Evaluation results": loads the saved 200-question experiment output
    and displays the metric tables/charts required by the project brief.

Run with: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CFG, PATHS
from src.corrective.pipeline import run_pipeline
from src.data.schema import Question
from src.evaluation.metrics import evaluate_answer
from src.knowledge_graph.wikidata_client import WikidataClient

st.set_page_config(page_title="Corrective RAG + KG Fallback", layout="wide")


@st.cache_resource
def get_client():
    return WikidataClient()


def render_demo_tab():
    st.header("Try a question")
    st.caption(
        "Enter any factual question. The pipeline retrieves evidence, grades it, "
        "and falls back to a live Wikidata lookup when the evidence is judged "
        "insufficient."
    )

    default_q = "who founded SpaceX?"
    question_text = st.text_input("Question", value=default_q)
    ground_truth = st.text_input("Ground truth answer (optional, for correctness check)", value="")
    run = st.button("Run pipeline", type="primary")

    if not run or not question_text.strip():
        return

    q = Question(
        question_id="demo",
        question=question_text.strip(),
        answer=ground_truth.strip(),
        question_type="unknown",
        domain="unknown",
        dynamism="unknown",
    )
    with st.spinner("Running retrieval -> grading -> (fallback) -> generation..."):
        client = get_client()
        result = run_pipeline(q, client)

    st.subheader("1. Retrieval")
    if result.retrieved_titles:
        st.write("Top retrieved passages (TF-IDF cosine similarity):")
        st.table(pd.DataFrame({
            "Title": result.retrieved_titles,
            "Score": result.retrieval_scores,
        }))
        with st.expander("Retrieved evidence text"):
            st.text(result.retrieval_context)
    else:
        st.warning(
            "No passages retrieved. Note: this demo's retrieval only works for "
            "questions whose CRAG-provided search_results are available - for an "
            "arbitrary typed-in question there is no web corpus attached, so "
            "retrieval will usually come back empty and fallback should trigger."
        )

    st.subheader("2. Baseline RAG answer")
    st.success(result.baseline_answer)

    st.subheader("3. Evidence quality / fallback decision")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fallback triggered", "YES" if result.fallback_triggered else "NO")
    col2.metric("Entity coverage", f"{result.entity_coverage:.0%}")
    col3.metric("Retrieval score (top-1)", f"{result.retrieval_scores[0]:.3f}" if result.retrieval_scores else "0.000")
    if result.fallback_triggered:
        st.write("**Why fallback triggered:**", ", ".join(result.fallback_reasons) or "none")
        if result.missing_entities:
            st.write("**Missing entities (not found in retrieved text):**", ", ".join(result.missing_entities))

    if result.fallback_triggered:
        st.subheader("4. Knowledge Graph entity linking")
        st.write("Detected entity mentions and resolved Wikidata QIDs:")
        st.table(pd.DataFrame({
            "Mention": result.kg_entities,
            "Resolved QID": result.kg_qids + [None] * (len(result.kg_entities) - len(result.kg_qids)),
        }))

        st.subheader("5. Knowledge Graph facts retrieved")
        if result.kg_found and result.kg_triples:
            st.table(pd.DataFrame(result.kg_triples))
        else:
            st.warning("No usable KG facts found for the linked entities.")

        st.subheader("6. Final Corrective RAG answer")
        st.success(result.corrective_answer)
    else:
        st.info("Evidence was sufficient - Corrective RAG reuses the Baseline RAG answer unchanged.")

    if ground_truth.strip():
        st.subheader("Correctness (against the ground truth you entered)")
        base_eval = evaluate_answer(result.baseline_answer, ground_truth)
        corr_eval = evaluate_answer(result.corrective_answer, ground_truth)
        c1, c2 = st.columns(2)
        c1.metric("Baseline correct", "YES" if base_eval.correct else "NO", f"F1={base_eval.token_f1:.2f}")
        c2.metric("Corrective correct", "YES" if corr_eval.correct else "NO", f"F1={corr_eval.token_f1:.2f}")


def render_evaluation_tab():
    st.header("200-Question Evaluation Results")

    summary_path = PATHS.metrics_dir / "summary.json"
    if not summary_path.exists():
        st.error(
            "No evaluation results found yet. Run:\n\n"
            "```\npython scripts/run_evaluation.py\npython scripts/analyse_results.py\n```"
        )
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    overall = summary["overall"]
    confusion = summary["confusion"]
    by_type = pd.DataFrame(summary["by_type"])
    by_domain = pd.DataFrame(summary["by_domain"])

    st.subheader("Overall metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline accuracy", f"{overall['baseline_accuracy']:.1%}")
    c2.metric("Corrective accuracy", f"{overall['corrective_accuracy']:.1%}",
              f"{overall['absolute_improvement']:+.1%}")
    c3.metric("KG fallback rate", f"{overall['kg_fallback_rate']:.1%}",
              f"{overall['total_kg_fallback_count']} questions")
    c4.metric("Fallback success rate", f"{overall['fallback_success_rate']:.1%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Accuracy when fallback triggered", f"{overall['accuracy_when_fallback_triggered']:.1%}")
    c6.metric("Accuracy when fallback not triggered", f"{overall['accuracy_when_fallback_not_triggered']:.1%}")
    c7.metric("Baseline-wrong -> Corrective-correct", overall["num_baseline_wrong_corrective_correct"])

    st.subheader("Fallback effectiveness (confusion-style breakdown)")
    st.table(pd.DataFrame([confusion]).T.rename(columns={0: "count"}))

    plot_files = [
        ("01_overall_accuracy.png", "Overall accuracy: Baseline vs Corrective"),
        ("02_accuracy_by_type.png", "Accuracy by question type"),
        ("03_fallback_rate_by_type.png", "KG fallback rate by question type"),
        ("04_outcome_distribution.png", "Fallback outcome distribution"),
        ("05_baseline_vs_corrective.png", "Per-type baseline vs corrective"),
    ]
    st.subheader("Charts")
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(plot_files):
        fpath = PATHS.plots_dir / fname
        if fpath.exists():
            cols[i % 2].image(str(fpath), caption=caption, use_container_width=True)

    st.subheader("Per-question-type metrics")
    st.dataframe(by_type, use_container_width=True)

    st.subheader("Per-domain metrics")
    st.dataframe(by_domain, use_container_width=True)

    st.subheader("Representative examples")
    preds_path = PATHS.predictions_dir / "predictions.jsonl"
    if preds_path.exists():
        with open(preds_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        improved = [r for r in records if not r["baseline_correct"] and r["corrective_correct"]]
        worsened = [r for r in records if r["baseline_correct"] and not r["corrective_correct"]]

        tab_a, tab_b = st.tabs([f"Improved by fallback ({len(improved)})", f"Worsened by fallback ({len(worsened)})"])
        with tab_a:
            for r in improved[:10]:
                with st.expander(f"[{r['question_type']}] {r['question']}"):
                    st.write("Gold:", r["ground_truth"])
                    st.write("Baseline:", r["baseline_answer"])
                    st.write("Corrective:", r["corrective_answer"])
                    st.write("Fallback reasons:", ", ".join(r["fallback_reasons"]))
                    st.write("KG facts used:", r["kg_triples"])
        with tab_b:
            for r in worsened[:10]:
                with st.expander(f"[{r['question_type']}] {r['question']}"):
                    st.write("Gold:", r["ground_truth"])
                    st.write("Baseline:", r["baseline_answer"])
                    st.write("Corrective:", r["corrective_answer"])
                    st.write("Fallback reasons:", ", ".join(r["fallback_reasons"]))
                    st.write("KG facts used:", r["kg_triples"])


def main():
    st.title("Corrective RAG with Knowledge-Graph Fallback")
    st.caption("P3 - KGIRS Microproject | CRAG benchmark, 200-question evaluation")
    st.sidebar.info(
        f"LLM generation (Groq): {'ENABLED - ' + CFG.groq_model if CFG.use_llm else 'DISABLED (local extractive fallback active)'}"
    )
    tab1, tab2 = st.tabs(["Try a question", "Evaluation results"])
    with tab1:
        render_demo_tab()
    with tab2:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
