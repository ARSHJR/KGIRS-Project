"""Single source of truth for all project parameters.

Every tunable constant used across the pipeline lives here so the whole
experiment can be reproduced or re-tuned from one place.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    results_dir: Path = PROJECT_ROOT / "results"
    predictions_dir: Path = PROJECT_ROOT / "results" / "predictions"
    metrics_dir: Path = PROJECT_ROOT / "results" / "metrics"
    plots_dir: Path = PROJECT_ROOT / "results" / "plots"

    def ensure(self) -> None:
        for p in [self.raw_dir, self.processed_dir, self.cache_dir,
                  self.results_dir, self.predictions_dir, self.metrics_dir, self.plots_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    # --- dataset sampling ---
    sample_size: int = 200
    random_seed: int = 42
    raw_subset_files: tuple = (
        "crag_subset_1.jsonl.bz2",
        "crag_subset_2.jsonl.bz2",
    )
    eval_subset_filename: str = "eval_subset_200.jsonl"

    # --- retrieval ---
    top_k: int = 3
    retrieval_method: str = "tfidf"  # tfidf over CRAG-provided search_results

    # --- corrective / fallback gating ---
    # Evidence is graded on a 0-1 scale combining retrieval similarity and
    # question-entity coverage in the retrieved passages. If the combined
    # score falls below fallback_threshold, Corrective RAG triggers the
    # Knowledge-Graph lookup instead of trusting the RAG context.
    fallback_similarity_threshold: float = 0.12
    fallback_entity_coverage_threshold: float = 0.5
    min_passage_overlap_words: int = 3

    # --- knowledge graph ---
    wikidata_sparql_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "WIKIDATA_SPARQL_ENDPOINT", "https://query.wikidata.org/sparql"
        )
    )
    wikidata_api_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "WIKIDATA_API_ENDPOINT", "https://www.wikidata.org/w/api.php"
        )
    )
    wikidata_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "WIKIDATA_USER_AGENT", "CorrectiveRAG-CollegeProject/1.0"
        )
    )
    kg_cache_filename: str = "wikidata_cache.json"
    kg_max_candidate_entities: int = 3
    kg_request_timeout_sec: int = 8

    # --- answer generation (LLM, optional) ---
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_api_base: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    )
    llm_max_tokens: int = 64
    llm_temperature: float = 0.0
    llm_timeout_sec: int = 20

    @property
    def use_llm(self) -> bool:
        return bool(self.groq_api_key)


PATHS = Paths()
CFG = Config()
