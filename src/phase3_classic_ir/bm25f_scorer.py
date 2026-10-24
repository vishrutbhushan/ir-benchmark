"""
BM25F Scorer — Hand-coded Zone-Weighted BM25F Scoring with Hierarchical Conditioning.

Implements the standard BM25F ranking function from scratch:

    tf_tilde(t, d) = SUM over zone z:
        w_z * ( tf_z(t, d) / ( (1 - b_z) + b_z * (|d_z| / avgdl_z) ) )

    Score_BM25F(q, d) = SUM over t in q:
        IDF(t) * ( tf_tilde(t, d) / (k1 + tf_tilde(t, d)) )

Where:
    - w_z: weight of zone z (e.g., title=4.0, heading=2.5, code=2.0, body=1.0)
    - b_z: length normalization parameter per zone z
    - |d_z|: length of document d in zone z
    - avgdl_z: corpus-wide average length of zone z
    - k1: term frequency saturation parameter (default 1.2)
    - IDF(t): Robertson-Spärck Jones IDF

Hierarchical Conditioning (Optional):
    If hierarchical boosting is enabled and query taxonomy inference detects an explicit
    technology or category in the query, documents belonging to that branch of the
    hierarchy receive a principled relevance boost.

Reference: Robertson et al. (2004), "Simple BM25 Extension to Multiple Weighted Fields"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.phase3_classic_ir.inverted_index import InvertedIndex, ZONES
from src.utils.taxonomy import infer_query_taxonomy


DEFAULT_ZONE_WEIGHTS: dict[str, float] = {
    "title": 4.0,
    "heading": 2.5,
    "code": 2.0,
    "body": 1.0,
}

DEFAULT_ZONE_B: dict[str, float] = {
    "title": 0.3,
    "heading": 0.5,
    "code": 0.6,
    "body": 0.75,
}


class BM25FScorer:
    """
    Hand-coded zone-weighted BM25F scorer operating over an InvertedIndex.
    """

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.2,
        zone_weights: dict[str, float] | None = None,
        zone_b: dict[str, float] | None = None,
        hierarchical_boost: float = 0.25,
    ) -> None:
        """
        Args:
            index: Pre-built inverted index with per-zone data.
            k1: Term frequency saturation parameter.
            zone_weights: Weight per zone w_z (defaults: title=4.0, heading=2.5, code=2.0, body=1.0).
            zone_b: Length normalization parameter b_z per zone.
            hierarchical_boost: Bonus factor applied to documents whose category/technology
                                matches the query's inferred taxonomy intent (0.0 to disable).
        """
        self.index = index
        self.k1 = k1
        self.zone_weights = dict(DEFAULT_ZONE_WEIGHTS)
        if zone_weights:
            self.zone_weights.update(zone_weights)

        self.zone_b = dict(DEFAULT_ZONE_B)
        if zone_b:
            self.zone_b.update(zone_b)

        self.hierarchical_boost = max(0.0, hierarchical_boost)

    def compute_pseudo_tf(self, doc_id: str, zone_tfs: dict[str, int]) -> float:
        """
        Computes the BM25F normalized pseudo-frequency tf_tilde(t, d)
        by aggregating normalized frequencies across all active zones.
        """
        pseudo_tf = 0.0

        for zone in ZONES:
            tf_z = zone_tfs.get(zone, 0)
            if tf_z <= 0:
                continue

            w_z = self.zone_weights.get(zone, 1.0)
            b_z = self.zone_b.get(zone, 0.75)
            len_z = self.index.get_zone_length(doc_id, zone)
            avg_z = self.index.avg_zone_lengths.get(zone, 0.0)

            # Normalization factor B_z
            b_factor = (1.0 - b_z) + b_z * (len_z / avg_z) if avg_z > 0 else 1.0
            if b_factor <= 0.0:
                b_factor = 1.0

            pseudo_tf += w_z * (tf_z / b_factor)

        return pseudo_tf

    def score_query(
        self,
        query_terms: list[str],
        query_raw_text: str = "",
        top_k: int = 20,
        category_filter: str | None = None,
        routing_mode: str = "hybrid",
        use_tiered: bool = False,
        tier1_threshold: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Scores candidate documents against query terms using the BM25F ranking formula,
        supporting multi-level category routing, tiered champion list pruning, and zone weights.

        Args:
            query_terms: Tokenized and stemmed query terms.
            query_raw_text: Original raw query string (used for hierarchical taxonomy inference).
            top_k: Number of top results to return.
            category_filter: Optional hard restriction to a specific category.
            routing_mode: "hybrid" (soft boost), "hierarchical_first" (category-first sub-index search),
                          or "tiered" (champion list pruning).
            use_tiered: If True, uses Tier-1 champion lists first to prune candidate space.
            tier1_threshold: Minimum candidates needed to avoid evaluating Tier-2 postings.

        Returns:
            Ranked list of top-k retrieved documents with scores and metadata.
        """
        # Pre-infer taxonomy if query text is provided
        query_tax = infer_query_taxonomy(query_raw_text) if query_raw_text else {}
        target_techs = set(query_tax.get("matched_technologies", []))
        target_cats = set(query_tax.get("matched_categories", []))

        # Hierarchical-First Routing: if enabled, attempt category sub-index search first
        if routing_mode == "hierarchical_first" and not category_filter and target_cats:
            primary_cat = list(target_cats)[0]
            cat_results = self.score_query(
                query_terms=query_terms,
                query_raw_text=query_raw_text,
                top_k=top_k,
                category_filter=primary_cat,
                routing_mode="hybrid",
            )
            # If category search found enough candidates, return them immediately
            if len(cat_results) >= min(top_k, 5):
                return cat_results
            # Otherwise fall through to global search for recall safety

        scores: dict[str, float] = defaultdict(float)

        # Restrict candidate pool if category_filter is specified
        allowed_docs: set[str] | None = None
        if category_filter:
            allowed_docs = set(self.index.get_docs_in_category(category_filter))
            if not allowed_docs:
                return []

        # Determine if tiered champion pruning is active
        run_tiered = use_tiered or (routing_mode == "tiered")

        def _evaluate_postings(posting_getter) -> int:
            matched_candidates: set[str] = set()
            for term in query_terms:
                idf = self.index.idf(term)
                if idf <= 0.0:
                    continue

                for posting in posting_getter(term):
                    doc_id = posting.doc_id
                    if allowed_docs is not None and doc_id not in allowed_docs:
                        continue

                    pseudo_tf = self.compute_pseudo_tf(doc_id, posting.zone_tfs)
                    if pseudo_tf <= 0.0:
                        continue

                    term_score = idf * (pseudo_tf / (self.k1 + pseudo_tf))
                    scores[doc_id] += term_score
                    matched_candidates.add(doc_id)
            return len(matched_candidates)

        if run_tiered:
            # Pass 1: Evaluate Tier-1 Champion lists
            num_tier1 = _evaluate_postings(self.index.get_champion_postings)
            # Pass 2: If Tier-1 produced fewer than tier1_threshold candidates, include Tier-2
            if num_tier1 < tier1_threshold:
                _evaluate_postings(self.index.get_tier2_postings)
        else:
            _evaluate_postings(self.index.get_postings)

        if not scores:
            return []

        # Apply Hierarchical Taxonomy Conditioning (soft boost if document matches inferred taxonomy)
        final_scores: list[tuple[str, float, dict[str, Any]]] = []
        for doc_id, base_score in scores.items():
            meta = self.index.doc_metadata.get(doc_id, {})
            doc_tech = meta.get("technology", "")
            doc_cat = meta.get("category", "")

            boost_multiplier = 1.0
            if self.hierarchical_boost > 0:
                if doc_tech and doc_tech in target_techs:
                    # Direct technology match (strongest hierarchical signal)
                    boost_multiplier += self.hierarchical_boost
                elif doc_cat and doc_cat in target_cats:
                    # Category match (medium hierarchical signal)
                    boost_multiplier += (self.hierarchical_boost * 0.5)

            final_score = base_score * boost_multiplier
            final_scores.append((doc_id, final_score, meta))

        # Sort candidate documents by descending score
        final_scores.sort(key=lambda x: -x[1])

        # Return top-k with metadata
        results: list[dict[str, Any]] = []
        for rank, (doc_id, s, meta) in enumerate(final_scores[:top_k], start=1):
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(s, 6),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "technology": meta.get("technology", ""),
                "category": meta.get("category", ""),
                "domain": meta.get("domain", ""),
            })

        return results

    def score_query_multi_level(
        self,
        query_terms: list[str],
        query_raw_text: str = "",
        top_k_docs: int = 10,
        top_k_sections: int = 5,
        routing_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """
        Simultaneously performs multi-level document and passage/section retrieval.
        Returns both top-ranked documents and their most relevant matching sections.
        """
        docs = self.score_query(
            query_terms=query_terms,
            query_raw_text=query_raw_text,
            top_k=top_k_docs,
            routing_mode=routing_mode,
        )
        top_doc_ids = {d["doc_id"] for d in docs}
        sections = self.index.score_sections(
            query_terms=query_terms,
            top_k=top_k_sections,
            doc_id_filter=top_doc_ids,
        )
        return {
            "documents": docs,
            "sections": sections,
        }

