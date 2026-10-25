"""Thin, cached client for the two real Wikidata access paths used by the
Knowledge-Graph fallback:

  1. wbsearchentities API  - resolves a free-text entity mention to
     candidate Wikidata QIDs (entity linking).
  2. SPARQL query service  - given a QID, retrieves its direct property/
     value claims as labelled (subject, predicate, object) triples.

Every successful response is cached to disk (data/cache/wikidata_cache.json)
keyed by request type + normalised input, so repeated evaluation runs do
not re-hit the network and stay reproducible. Network/API errors are
caught and surfaced as an explicit empty result rather than raising, so a
single flaky lookup cannot crash a 200-question run.
"""
import json
import time
from pathlib import Path
from typing import List, Optional

import requests

from src.config import CFG, PATHS


class WikidataClient:
    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or (PATHS.cache_dir / CFG.kg_cache_filename)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self.headers = {"User-Agent": CFG.wikidata_user_agent}

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _cache_get(self, key: str):
        return self._cache.get(key)

    def _cache_set(self, key: str, value) -> None:
        self._cache[key] = value
        self._save_cache()

    # ---------------- entity linking ----------------
    def search_entity(self, text: str, limit: int = 3) -> List[dict]:
        """Resolve free text to candidate Wikidata entities via
        wbsearchentities. Returns [] on error or no match (never raises)."""
        key = f"search::{text.strip().lower()}::{limit}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        params = {
            "action": "wbsearchentities",
            "search": text,
            "language": "en",
            "format": "json",
            "limit": limit,
            "type": "item",
        }
        try:
            resp = requests.get(
                CFG.wikidata_api_endpoint,
                params=params,
                headers=self.headers,
                timeout=CFG.kg_request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            results = [
                {
                    "id": item["id"],
                    "label": item.get("label", ""),
                    "description": item.get("description", ""),
                }
                for item in data.get("search", [])
            ]
        except (requests.RequestException, ValueError, KeyError):
            results = []

        self._cache_set(key, results)
        return results

    # ---------------- triple retrieval ----------------
    def get_entity_triples(self, qid: str, limit: int = 30) -> List[dict]:
        """Run a SPARQL query retrieving the entity's direct claims as
        labelled (property, value) pairs. Returns [] on error/no results."""
        key = f"triples::{qid}::{limit}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        query = f"""
        SELECT ?propLabel ?valueLabel WHERE {{
          wd:{qid} ?p ?value .
          ?prop wikibase:directClaim ?p .
          ?prop rdfs:label ?propLabel . FILTER(LANG(?propLabel)="en")
          OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel)="en") }}
          FILTER(!ISBLANK(?value))
        }}
        LIMIT {limit}
        """
        try:
            resp = requests.get(
                CFG.wikidata_sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={**self.headers, "Accept": "application/sparql-results+json"},
                timeout=CFG.kg_request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            triples = []
            for b in data.get("results", {}).get("bindings", []):
                prop_label = b.get("propLabel", {}).get("value", "")
                value = b.get("valueLabel", {}).get("value") or b.get("value", {}).get("value", "")
                if prop_label and value:
                    triples.append({"property": prop_label, "value": value})
        except (requests.RequestException, ValueError, KeyError):
            triples = []

        self._cache_set(key, triples)
        return triples

    def get_entity_label(self, qid: str) -> str:
        key = f"label::{qid}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        try:
            resp = requests.get(
                CFG.wikidata_api_endpoint,
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
                headers=self.headers,
                timeout=CFG.kg_request_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            label = data["entities"][qid]["labels"]["en"]["value"]
        except (requests.RequestException, ValueError, KeyError):
            label = qid
        self._cache_set(key, label)
        return label
