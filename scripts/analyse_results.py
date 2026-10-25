"""Turn results/predictions/predictions.jsonl into every metric table and
chart required by the project brief, written to results/metrics/*.csv and
results/plots/*.png.

Run after scripts/run_evaluation.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PATHS
from src.evaluation.aggregate import (
    confusion_breakdown,
    overall_metrics,
    per_domain_metrics,
    per_dynamism_metrics,
    per_question_type_metrics,
)

# --- palette (validated categorical order, light-surface chart chrome) ---
BLUE = "#2a78d6"     # baseline RAG
ORANGE = "#eb6834"   # corrective RAG
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "font.size": 10,
})


def load_records():
    path = PATHS.predictions_dir / "predictions.jsonl"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def chart_overall_accuracy(overall: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["Baseline RAG", "Corrective RAG"]
    values = [overall["baseline_accuracy"], overall["corrective_accuracy"]]
    bars = ax.bar(labels, values, color=[BLUE, ORANGE], width=0.55)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.1%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy: Baseline vs Corrective RAG", color=TEXT_PRIMARY, fontsize=12)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_accuracy_by_type(rows: list, out_path: Path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["baseline_accuracy"], width, label="Baseline RAG", color=BLUE)
    ax.bar([i + width / 2 for i in x], df["corrective_accuracy"], width, label="Corrective RAG", color=ORANGE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["question_type"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Question Type", color=TEXT_PRIMARY, fontsize=12)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_fallback_rate_by_type(rows: list, out_path: Path):
    df = pd.DataFrame(rows).sort_values("kg_fallback_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(df))]
    bars = ax.barh(df["question_type"], df["kg_fallback_rate"], color=colors)
    for b, v in zip(bars, df["kg_fallback_rate"]):
        ax.annotate(f"{v:.0%}", (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xlabel("KG Fallback Rate")
    ax.set_title("KG Fallback Rate by Question Type", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlim(0, 1.05)
    _style_axes(ax)
    ax.grid(axis="x", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_outcome_distribution(confusion: dict, out_path: Path):
    labels = [
        "Baseline correct\n& Corrective correct",
        "Baseline wrong\n& Corrective correct\n(improved)",
        "Baseline correct\n& Corrective wrong\n(worsened)",
        "Baseline wrong\n& Corrective wrong",
    ]
    keys = [
        "baseline_correct_corrective_correct",
        "baseline_wrong_corrective_correct",
        "baseline_correct_corrective_wrong",
        "baseline_wrong_corrective_wrong",
    ]
    values = [confusion[k] for k in keys]
    colors = [AQUA, GREEN, RED, TEXT_SECONDARY]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for b, v in zip(bars, values):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("Number of Questions")
    ax.set_title("Fallback Outcome Distribution (Baseline vs Corrective)", color=TEXT_PRIMARY, fontsize=12)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chart_baseline_vs_corrective_scatter(rows: list, out_path: Path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["baseline_accuracy"], df["corrective_accuracy"],
               s=df["num_questions"] * 6, color=BLUE, alpha=0.75, edgecolor=TEXT_PRIMARY, linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["question_type"], (row["baseline_accuracy"], row["corrective_accuracy"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    lims = [0, 1]
    ax.plot(lims, lims, linestyle="--", color=AXIS, linewidth=1, label="No change (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Baseline RAG Accuracy")
    ax.set_ylabel("Corrective RAG Accuracy")
    ax.set_title("Per-Type Accuracy: Baseline vs Corrective\n(bubble size = # questions)", color=TEXT_PRIMARY, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    PATHS.ensure()
    records = load_records()
    print(f"Loaded {len(records)} prediction records.")

    overall = overall_metrics(records)
    confusion = confusion_breakdown(records)
    by_type = per_question_type_metrics(records)
    by_domain = per_domain_metrics(records)
    by_dynamism = per_dynamism_metrics(records)

    print(json.dumps(overall, indent=2))
    print(json.dumps(confusion, indent=2))

    # --- CSV tables ---
    pd.DataFrame([overall]).to_csv(PATHS.metrics_dir / "overall_metrics.csv", index=False)
    pd.DataFrame([confusion]).to_csv(PATHS.metrics_dir / "confusion_breakdown.csv", index=False)
    pd.DataFrame(by_type).to_csv(PATHS.metrics_dir / "per_question_type_metrics.csv", index=False)
    pd.DataFrame(by_domain).to_csv(PATHS.metrics_dir / "per_domain_metrics.csv", index=False)
    pd.DataFrame(by_dynamism).to_csv(PATHS.metrics_dir / "per_dynamism_metrics.csv", index=False)

    with open(PATHS.metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "confusion": confusion, "by_type": by_type,
                    "by_domain": by_domain, "by_dynamism": by_dynamism},
                   f, indent=2)

    # --- charts ---
    chart_overall_accuracy(overall, PATHS.plots_dir / "01_overall_accuracy.png")
    chart_accuracy_by_type(by_type, PATHS.plots_dir / "02_accuracy_by_type.png")
    chart_fallback_rate_by_type(by_type, PATHS.plots_dir / "03_fallback_rate_by_type.png")
    chart_outcome_distribution(confusion, PATHS.plots_dir / "04_outcome_distribution.png")
    chart_baseline_vs_corrective_scatter(by_type, PATHS.plots_dir / "05_baseline_vs_corrective.png")

    print(f"\nCSV tables written to {PATHS.metrics_dir}")
    print(f"Charts written to {PATHS.plots_dir}")


if __name__ == "__main__":
    main()
