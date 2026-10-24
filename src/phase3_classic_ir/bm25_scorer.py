"""
BM25 Scorer — Hand-coded BM25 scoring function.

Implements the standard Okapi BM25 ranking function from scratch:

    score(q, d) = SUM over t in q:
        IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

Where:
    - IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    - tf(t,d) = combined term frequency across all zones (flat BM25)
    - |d| = total document length (all zones combined)
    - avgdl = average document length across corpus
    - k1 = term frequency saturation parameter (default 1.2)
    - b = document length normalization parameter (default 0.75)

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"

No off-the-shelf retrieval libraries are used (project constraint).
"""

from __future__ import annotations

from typing import Any

from src.phase3_classic_ir.inverted_index import InvertedIndex, ZONES


class BM25Scorer:
    """
    Hand-coded BM25 scorer operating on a pre-built InvertedIndex.

    This is the flat (single-field) variant: all zone TFs are summed into
    one combined TF, and document length is the sum of all zone lengths.
    """

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """
        Args:
            index: The pre-built inverted index.
            k1: Term frequency saturation parameter. Higher values give
                more weight to repeated term occurrences. Typical: 1.2–2.0.
            b: Document length normalization. 0 = no normalization,
                1 = full normalization relative to average. Typical: 0.75.
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self.avgdl = index.get_avg_total_doc_length()

    def score_document(self, query_terms: list[str], doc_id: str) -> float:
        """
        Compute BM25 score for a single document against query terms.

        Args:
            query_terms: Tokenized and stemmed query terms.
            doc_id: The document to score.

        Returns:
            BM25 relevance score (higher = more relevant).
        """
        doc_len = self.index.get_total_doc_length(doc_id)
        score = 0.0

        for term in query_terms:
            # Get IDF for this term
            idf = self.index.idf(term)
            if idf == 0.0:
                continue

            # Get combined TF across all zones (flat BM25)
            postings = self.index.get_postings(term)
            tf = 0
            for posting in postings:
                if posting.doc_id == doc_id:
                    tf = sum(posting.zone_tfs.values())
                    break

            if tf == 0:
                continue

            # BM25 formula
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def score_query(
        self,
        query_terms: list[str],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Score all documents in the index against the query and return
        the top-k ranked results.

        Args:
            query_terms: Tokenized and stemmed query terms.
            top_k: Number of top results to return.

        Returns:
            List of dicts with doc_id, score, and metadata, sorted by
            descending score.
        """
        # Collect candidate documents: union of all docs containing any query term
        candidate_docs: set[str] = set()
        for term in query_terms:
            for posting in self.index.get_postings(term):
                candidate_docs.add(posting.doc_id)

        if not candidate_docs:
            return []

        # Score each candidate
        scored: list[tuple[str, float]] = []
        for doc_id in candidate_docs:
            s = self.score_document(query_terms, doc_id)
            if s > 0:
                scored.append((doc_id, s))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Return top-k with metadata
        results: list[dict[str, Any]] = []
        for doc_id, s in scored[:top_k]:
            meta = self.index.doc_metadata.get(doc_id, {})
            results.append({
                "rank": len(results) + 1,
                "doc_id": doc_id,
                "score": round(s, 6),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "technology": meta.get("technology", ""),
                "category": meta.get("category", ""),
            })

        return results
