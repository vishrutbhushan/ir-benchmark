"""
Query Processor — Handles query-side tokenization and search execution.

Applies the same tokenization pipeline used for document indexing to the
query text, then dispatches to the appropriate scorer (BM25 or BM25F).
"""

from __future__ import annotations

import collections
import json
import logging
import threading
from pathlib import Path
from typing import Any

from src.classical_preprocessing.tokenizer import tokenize
from src.retrieval.bm25_scorer import BM25Scorer
from src.retrieval.inverted_index import InvertedIndex


class LRUQueryCache:
    """Thread-safe generic Least-Recently-Used (LRU) query result cache."""

    def __init__(self, maxsize: int = 1024) -> None:
        self.maxsize = maxsize
        self._cache: collections.OrderedDict[tuple[str, int], dict[str, Any]] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[str, int]) -> dict[str, Any] | None:
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                # Return deep copy of results list to avoid caller mutation
                val = dict(self._cache[key])
                val["cache_hit"] = True
                return val
            self.misses += 1
            return None

    def put(self, key: tuple[str, int], value: dict[str, Any]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }


class QueryProcessor:
    """
    Processes raw query text through the shared tokenizer and executes
    search against the inverted index using BM25.
    """

    def __init__(
        self,
        index: InvertedIndex,
        scorer: Any,
        logger: logging.Logger | None = None,
        routing_mode: str = "hybrid",
        cache_size: int = 1024,
    ) -> None:
        self.index = index
        self.scorer = scorer
        self.logger = logger or logging.getLogger(__name__)
        self.routing_mode = routing_mode
        self.cache = LRUQueryCache(maxsize=cache_size)

    def process_query(self, query_text: str) -> list[str]:
        """
        Tokenize a raw query string using the same pipeline as documents.

        Returns the list of stemmed/normalized query tokens.
        """
        return tokenize(query_text, stem=True, remove_stopwords=True, expand_identifiers=True)

    def search(
        self,
        query_text: str,
        top_k: int = 20,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a full search: tokenize query → score all docs → return top-k.
        Uses LRU caching to return instant results on repeated queries.

        Args:
            query_text: Raw query string.
            top_k: Number of results to return.
            use_cache: Whether to check and populate the LRU cache.

        Returns:
            Dict with query info, tokens, ranked results, and cache_hit flag.
        """
        cache_key = (query_text.strip().lower(), top_k)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        query_tokens = self.process_query(query_text)
        self.logger.info(f"Query: \"{query_text}\" -> tokens: {query_tokens}")

        import inspect
        sig = inspect.signature(self.scorer.score_query)
        kwargs: dict[str, Any] = {"top_k": top_k}
        if "query_raw_text" in sig.parameters:
            kwargs["query_raw_text"] = query_text
        if "routing_mode" in sig.parameters:
            kwargs["routing_mode"] = self.routing_mode

        results = self.scorer.score_query(query_tokens, **kwargs)

        payload = {
            "query_text": query_text,
            "query_tokens": query_tokens,
            "num_results": len(results),
            "results": results,
            "cache_hit": False,
        }

        if use_cache:
            self.cache.put(cache_key, payload)

        return payload

    def run_query_suite(
        self,
        queries_path: Path,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Run all queries from a JSONL query file against the index.

        Args:
            queries_path: Path to queries.jsonl with fields:
                query_id, query_text, category
            top_k: Number of results per query.

        Returns:
            List of search result dicts, one per query.
        """
        all_results: list[dict[str, Any]] = []

        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                query_record = json.loads(line)
                query_id = query_record["query_id"]
                query_text = query_record["query_text"]
                query_category = query_record.get("category", "")

                self.logger.info(f"Running {query_id}: \"{query_text[:60]}...\"")

                search_result = self.search(query_text, top_k=top_k)
                search_result["query_id"] = query_id
                search_result["query_category"] = query_category

                self.logger.info(
                    f"  {query_id}: {search_result['num_results']} results "
                    f"(top score: {search_result['results'][0]['score'] if search_result['results'] else 0})"
                )

                all_results.append(search_result)

        return all_results
