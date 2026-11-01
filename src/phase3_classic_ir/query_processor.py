"""
Query Processor — Handles query-side tokenization and search execution.

Applies the same tokenization pipeline used for document indexing to the
query text, then dispatches to the appropriate scorer (BM25 or BM25F).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.phase2_tokenization.tokenizer import tokenize
from src.phase3_classic_ir.bm25_scorer import BM25Scorer
from src.phase3_classic_ir.inverted_index import InvertedIndex


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Hand-coded Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class QueryProcessor:
    """
    Processes raw query text through the shared tokenizer and executes
    search against the inverted index using BM25 / BM25F.
    Includes hand-coded Boolean operator cleanup and OOV/rare typo spell resolution.
    """

    def __init__(
        self,
        index: InvertedIndex,
        scorer: Any,
        logger: logging.Logger | None = None,
        routing_mode: str = "hybrid",
    ) -> None:
        self.index = index
        self.scorer = scorer
        self.logger = logger or logging.getLogger(__name__)
        self.routing_mode = routing_mode

        # Build fast lookup buckets (first_char, length) -> [(term, df)] for spell fallback
        self._spell_buckets: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        min_df = max(2, min(15, index.total_docs // 500)) if index.total_docs > 10 else 1
        for term, plist in index.postings.items():
            df = len(plist)
            if df >= min_df and len(term) >= 3 and term.isalpha():
                self._spell_buckets[(term[0], len(term))].append((term, df))

    def _correct_typo_token(self, token: str) -> str:
        """
        If a token is missing from the vocabulary or has negligible DF (< 5 in large index),
        find the closest vocabulary term within Levenshtein edit distance <= 2.
        """
        df = self.index.document_frequency(token)
        threshold = 5 if self.index.total_docs > 1000 else 1
        if df >= threshold or len(token) < 4 or not token.isalpha():
            return token

        first_ch = token[0]
        t_len = len(token)
        best_term = token
        best_dist = 3
        best_df = df

        for cand_len in range(max(3, t_len - 2), t_len + 3):
            for cand_term, cand_df in self._spell_buckets.get((first_ch, cand_len), ()):
                dist = _levenshtein_distance(token, cand_term)
                if dist <= 2:
                    if dist < best_dist or (dist == best_dist and cand_df > best_df):
                        best_dist = dist
                        best_term = cand_term
                        best_df = cand_df

        return best_term

    def process_query(self, query_text: str) -> list[str]:
        """
        Tokenize a raw query string using the same pipeline as documents,
        stripping Boolean operators (AND/OR/NOT) and resolving OOV/typo tokens.

        Returns the list of stemmed/normalized query tokens.
        """
        # Strip explicit uppercase Boolean operators and clauses following NOT
        cleaned = re.sub(r"\bNOT\s+\w+(?:\s+\w+)?", " ", query_text)
        cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned)
        raw_tokens = tokenize(cleaned, stem=True, remove_stopwords=True, expand_identifiers=True)
        return [self._correct_typo_token(tok) for tok in raw_tokens]

    def search(
        self,
        query_text: str,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """
        Execute a full search: tokenize query → score all docs → return top-k.

        Args:
            query_text: Raw query string.
            top_k: Number of results to return.

        Returns:
            Dict with query info, tokens, and ranked results.
        """
        query_tokens = self.process_query(query_text)
        self.logger.info(f"Query: \"{query_text}\" -> tokens: {query_tokens}")

        import inspect
        sig = inspect.signature(self.scorer.score_query)
        kwargs: dict[str, Any] = {"top_k": top_k}
        if "query_raw_text" in sig.parameters:
            kwargs["query_raw_text"] = f"{query_text} {' '.join(query_tokens)}"
        if "routing_mode" in sig.parameters:
            kwargs["routing_mode"] = self.routing_mode

        results = self.scorer.score_query(query_tokens, **kwargs)

        return {
            "query_text": query_text,
            "query_tokens": query_tokens,
            "num_results": len(results),
            "results": results,
        }

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
