"""
Multi-Level Index — Dedicated Two-Tier & Hierarchical Index for Classical IR.

This module implements the complete Multi-Level Indexing architecture described
in Sections 3.3, 3.4, 4.2, and 4.3 of the Technical IR project proposal:

1. LEVEL 1 (Document-Level Zonal Index):
   - Inverted postings tracking term frequencies across document zones:
     title (w=4.0), heading (w=2.5), code (w=2.0), body (w=1.0).
   - Document length normalization per zone.

2. LEVEL 2 (Section / Passage-Level Sub-Index):
   - Indexes structural sections/passages (doc_id#sec_i) with heading and content.
   - Enables fine-grained passage retrieval and answer snippet extraction.

3. HIERARCHICAL TAXONOMY DIRECTORY (Domain -> Category -> Technology):
   - Fast posting lists partitioned by technical domain, category, and technology.
   - Supports Category-First routing and hierarchical score boosting.

4. TIERED CHAMPION LISTS (Postings Pruning):
   - Tier 1: Champion postings (title/heading matches & highest term frequencies).
   - Tier 2: Standard body mentions.
   - Prunes postings traversal to evaluate high-quality candidates first.

No third-party retrieval/vector libraries used (hand-coded from scratch).
"""

from __future__ import annotations

import json
import math
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, NamedTuple

from src.phase3_classic_ir.inverted_index import InvertedIndex, Posting, ZONES
from src.utils.taxonomy import get_taxonomy_for_tech, infer_query_taxonomy


class SectionPosting(NamedTuple):
    """Posting entry in the Level-2 Section/Passage index."""
    sec_id: str
    doc_id: str
    heading: str
    tf: int


class MultiLevelIndex:
    """
    Dedicated Multi-Level Index encapsulating Document-level, Section-level,
    and Hierarchical Category retrieval.
    """

    def __init__(self, base_index: InvertedIndex | None = None) -> None:
        self.base_index: InvertedIndex = base_index or InvertedIndex()

        # Level 1: Document Zonal Index references
        self.doc_zone_lengths = self.base_index.doc_zone_lengths
        self.avg_zone_lengths = self.base_index.avg_zone_lengths
        self.doc_metadata = self.base_index.doc_metadata

        # Level 2: Section / Passage Sub-Index
        self.section_postings: dict[str, list[SectionPosting]] = defaultdict(list)
        self.section_catalog: dict[str, dict[str, Any]] = {}
        self.section_lengths: dict[str, int] = {}
        self.avg_section_length: float = 0.0

        # Hierarchical Directory (Domain -> Category -> Technology)
        self.category_postings = self.base_index.category_postings
        self.technology_postings = self.base_index.technology_postings
        self.domain_postings = self.base_index.domain_postings

        # Tiered Champion Lists
        self.champion_postings = self.base_index.champion_postings
        self.tier2_postings = self.base_index.tier2_postings

        # Sync base index sections if already built
        if self.base_index.doc_sections:
            self._sync_from_base()

    def _sync_from_base(self) -> None:
        """Sync Level-2 section structures from base inverted index."""
        self.section_lengths = dict(self.base_index.section_lengths)
        self.avg_section_length = self.base_index.avg_section_length
        for did, secs in self.base_index.doc_sections.items():
            for s in secs:
                self.section_catalog[s["sec_id"]] = {
                    "doc_id": did,
                    "heading": s.get("heading", ""),
                    "level": s.get("level", 2),
                    "snippet": s.get("snippet", ""),
                }
        for term, pairs in self.base_index.section_postings.items():
            self.section_postings[term] = [
                SectionPosting(
                    sec_id=sec_id,
                    doc_id=sec_id.split("#")[0],
                    heading=self.section_catalog.get(sec_id, {}).get("heading", ""),
                    tf=tf,
                )
                for sec_id, tf in pairs
            ]

    # ── Level 1: Document Retrieval ──────────────────────────────────────────

    def search_documents(
        self,
        query_terms: list[str],
        query_raw_text: str = "",
        top_k: int = 20,
        k1: float = 1.2,
        hierarchical_boost: float = 0.25,
        routing_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Level 1: Retrieve top matching whole documents using zone-weighted BM25F
        and hierarchical category conditioning.
        """
        from src.phase3_classic_ir.bm25f_scorer import BM25FScorer
        scorer = BM25FScorer(
            self.base_index,
            k1=k1,
            hierarchical_boost=hierarchical_boost,
        )
        return scorer.score_query(
            query_terms=query_terms,
            query_raw_text=query_raw_text,
            top_k=top_k,
            routing_mode=routing_mode,
        )

    # ── Level 2: Section / Passage Retrieval ─────────────────────────────────

    def search_sections(
        self,
        query_terms: list[str],
        top_k: int = 10,
        doc_id_filter: set[str] | list[str] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        """
        Level 2: Retrieve fine-grained sections/passages matching query terms,
        optionally constrained to specific candidate documents.
        """
        return self.base_index.score_sections(
            query_terms=query_terms,
            top_k=top_k,
            doc_id_filter=doc_id_filter,
            k1=k1,
            b=b,
        )

    # ── Multi-Level Unified Retrieval (Document + Best Section) ─────────────

    def search_multilevel(
        self,
        query_terms: list[str],
        query_raw_text: str = "",
        top_k_docs: int = 10,
        top_k_sections_per_doc: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Two-Tier Multi-Level Search:
        First searches Level-1 to rank top documents, then queries Level-2 to extract
        the exact best-matching section/passage and snippet for each retrieved document.
        """
        top_docs = self.search_documents(
            query_terms=query_terms,
            query_raw_text=query_raw_text,
            top_k=top_k_docs,
        )

        results: list[dict[str, Any]] = []
        for d in top_docs:
            doc_id = d["doc_id"]
            # Retrieve best section within this document
            best_secs = self.search_sections(
                query_terms=query_terms,
                top_k=top_k_sections_per_doc,
                doc_id_filter={doc_id},
            )
            best_sec = best_secs[0] if best_secs else {}

            results.append({
                "rank": d["rank"],
                "doc_id": doc_id,
                "document_title": d["title"],
                "url": d["url"],
                "category": d["category"],
                "technology": d["technology"],
                "doc_score": d["score"],
                "best_section": {
                    "section_id": best_sec.get("section_id", ""),
                    "heading": best_sec.get("heading", ""),
                    "snippet": best_sec.get("snippet", ""),
                    "section_score": best_sec.get("score", 0.0),
                },
            })

        return results

    # ── Tiered Champion List Retrieval (Candidate Space Pruning) ─────────────

    def search_tiered(
        self,
        query_terms: list[str],
        top_k: int = 20,
        tier1_threshold: int = 20,
    ) -> dict[str, Any]:
        """
        Tiered search: searches Tier-1 Champion lists first.
        Returns search results along with candidate pruning metrics.
        """
        from src.phase3_classic_ir.bm25f_scorer import BM25FScorer
        scorer = BM25FScorer(self.base_index, k1=1.2)
        results = scorer.score_query(
            query_terms=query_terms,
            top_k=top_k,
            routing_mode="tiered",
            tier1_threshold=tier1_threshold,
        )
        return {
            "top_k": len(results),
            "results": results,
            "pruning_method": "Tier-1 Champion Lists",
        }
