"""
Inverted Index — Hand-coded multi-zone positional inverted index.

This module builds and manages an inverted index where each term maps to
a list of postings. Each posting contains:
    - doc_id: The document identifier
    - zone_tfs: Per-zone term frequency counts {zone_name: tf}
    - positions: List of (zone, position) tuples for phrase/proximity matching

The index also tracks per-document zone lengths and corpus-wide average
zone lengths needed for BM25/BM25F length normalization.

No off-the-shelf retrieval libraries are used (project constraint).
"""

from __future__ import annotations

import json
import math
import time
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple


from src.utils.taxonomy import get_taxonomy_for_tech

# Zone names used throughout the index
ZONES = ("title", "heading", "body", "code")


class Posting(NamedTuple):
    """A single posting entry in the inverted index."""
    doc_id: str
    zone_tfs: dict[str, int]          # only non-zero zones: {"body": 5, "heading": 1}
    positions: dict[str, list[int]]  # {"heading": [11], "body": [175, 190, ...]}


class InvertedIndex:
    """
    Hand-coded inverted index with per-zone term frequencies, positional
    information, and hierarchical category/technology postings.

    Attributes:
        postings: Mapping from term → list of Posting entries.
        doc_zone_lengths: Mapping from doc_id → {zone: token_count}.
        avg_zone_lengths: Corpus-wide average token count per zone.
        total_docs: Total number of documents indexed.
        doc_metadata: Mapping from doc_id → metadata dict (category, technology, url, etc.).
        vocabulary: Set of all unique terms in the index.
        category_postings: Mapping from category name → list of doc_ids.
        technology_postings: Mapping from technology name → list of doc_ids.
        domain_postings: Mapping from domain name → list of doc_ids.
    """

    def __init__(self) -> None:
        # term → list of Posting
        self.postings: dict[str, list[Posting]] = {}
        # doc_id → {zone: length}
        self.doc_zone_lengths: dict[str, dict[str, int]] = {}
        # {zone: avg_length}
        self.avg_zone_lengths: dict[str, float] = {}
        # Total indexed documents
        self.total_docs: int = 0
        # doc_id → metadata (category, technology, url, dates, etc.)
        self.doc_metadata: dict[str, dict[str, Any]] = {}
        # Complete vocabulary
        self.vocabulary: set[str] = set()
        # Hierarchical postings
        self.category_postings: dict[str, list[str]] = defaultdict(list)
        self.technology_postings: dict[str, list[str]] = defaultdict(list)
        self.domain_postings: dict[str, list[str]] = defaultdict(list)

        # Multi-level Tiered / Champion lists (Tier 1 = high importance, Tier 2 = remaining)
        self.champion_postings: dict[str, list[Posting]] = defaultdict(list)
        self.tier2_postings: dict[str, list[Posting]] = defaultdict(list)

        # Multi-level document -> section / passage hierarchy
        self.doc_sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.section_postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.section_lengths: dict[str, int] = {}
        self.avg_section_length: float = 0.0

    def build_from_tokenized_corpus(
        self,
        tokenized_path: Path,
        structured_path: Path,
        logger: logging.Logger,
    ) -> None:
        """
        Build the inverted index from Phase 2 tokenized corpus and Phase 1
        structured records (for metadata).

        Args:
            tokenized_path: Path to outputs/phase2/tokenized_corpus.jsonl
            structured_path: Path to outputs/phase1/structured_records.jsonl
            logger: Logger instance
        """
        logger.info("Building inverted index from tokenized corpus...")
        start_time = time.time()

        # Step 1: Load metadata from structured records
        logger.info(f"Loading document metadata from {structured_path}...")
        self._load_metadata(structured_path)

        # Step 2: Build postings from tokenized records
        logger.info(f"Processing tokenized records from {tokenized_path}...")

        # Temporary accumulator: term → {doc_id → {zone_tfs, positions}}
        temp_postings: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        zone_length_accum: dict[str, list[int]] = defaultdict(list)

        doc_count = 0
        with tokenized_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc_id = record["doc_id"]
                doc_count += 1

                # Track zone lengths for this document
                zone_lengths: dict[str, int] = {}
                for zone in ZONES:
                    tokens_key = f"{zone}_tokens"
                    tokens = record.get(tokens_key, [])
                    zone_lengths[zone] = len(tokens)
                    zone_length_accum[zone].append(len(tokens))

                    # Process each token: build zone TFs and positional info
                    for position, token in enumerate(tokens):
                        # Filter out corrupt/replacement tokens or empty tokens
                        token = token.replace("\ufffd", "").strip()
                        if not token or len(token) > 64:
                            continue

                        self.vocabulary.add(token)

                        if doc_id not in temp_postings[token]:
                            temp_postings[token][doc_id] = {
                                "zone_tfs": defaultdict(int),
                                "positions": defaultdict(list),
                            }

                        temp_postings[token][doc_id]["zone_tfs"][zone] += 1
                        temp_postings[token][doc_id]["positions"][zone].append(position)

                self.doc_zone_lengths[doc_id] = zone_lengths

                # Step 2b: Multi-level section postings
                for sec in record.get("sections", []):
                    sec_id = sec.get("sec_id", "")
                    sec_toks = sec.get("tokens", [])
                    if not sec_id or not sec_toks:
                        continue
                    self.section_lengths[sec_id] = len(sec_toks)
                    sec_tf: dict[str, int] = defaultdict(int)
                    for st in sec_toks:
                        st = st.replace("\ufffd", "").strip()
                        if st and len(st) <= 64:
                            sec_tf[st] += 1
                    for st, scnt in sec_tf.items():
                        self.section_postings[st].append((sec_id, scnt))

                if doc_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Indexed {doc_count} docs, {len(temp_postings):,} unique terms "
                        f"({elapsed:.1f}s)..."
                    )

        self.total_docs = doc_count

        # Step 3: Compute average zone lengths
        for zone in ZONES:
            lengths = zone_length_accum[zone]
            self.avg_zone_lengths[zone] = sum(lengths) / len(lengths) if lengths else 0.0

        if self.section_lengths:
            self.avg_section_length = sum(self.section_lengths.values()) / len(self.section_lengths)

        # Step 4: Convert temp_postings to final Posting objects
        logger.info("Finalizing postings lists...")
        for term, doc_entries in temp_postings.items():
            posting_list: list[Posting] = []
            for did, data in doc_entries.items():
                posting_list.append(
                    Posting(
                        doc_id=did,
                        zone_tfs=dict(data["zone_tfs"]),
                        positions=dict(data["positions"]),
                    )
                )
            # Sort postings by doc_id for deterministic output
            posting_list.sort(key=lambda p: p.doc_id)
            self.postings[term] = posting_list

        # Step 5: Build Tiered Champion Lists
        logger.info("Building Tier-1 Champion lists for multi-level candidate pruning...")
        self.build_champion_lists(top_r=50)

        total_time = time.time() - start_time
        logger.info(
            f"Inverted index built in {total_time:.1f}s. "
            f"Documents: {self.total_docs}, "
            f"Vocabulary: {len(self.vocabulary):,} terms, "
            f"Total postings entries: {sum(len(pl) for pl in self.postings.values()):,}, "
            f"Sections indexed: {len(self.section_lengths):,}"
        )
        logger.info(
            f"Avg zone lengths: "
            + ", ".join(f"{z}={self.avg_zone_lengths[z]:.1f}" for z in ZONES)
        )

    def _load_metadata(self, structured_path: Path) -> None:
        """Load document metadata from structured records and populate hierarchical postings."""
        with structured_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc_id = record["doc_id"]
                raw_tech = record.get("metadata", {}).get("technology", "")
                raw_cat = record.get("metadata", {}).get("category", "")

                # Resolve against 2-tier hierarchical taxonomy
                tax = get_taxonomy_for_tech(raw_tech)
                category = tax["category"] if tax["category"] != "Other" else (raw_cat or "Other")
                domain = tax["domain"]

                self.doc_metadata[doc_id] = {
                    "url": record.get("url", ""),
                    "title": record.get("title", ""),
                    "technology": raw_tech,
                    "category": category,
                    "domain": domain,
                    "dates": record.get("dates", []),
                }

                # Populate section catalog for multi-level snippet & passage retrieval
                sections = record.get("sections", [])
                if sections:
                    self.doc_sections[doc_id] = [
                        {
                            "sec_id": f"{doc_id}#sec_{idx}",
                            "heading": s.get("heading", ""),
                            "level": s.get("level", 2),
                            "snippet": (s.get("content", "") or "")[:240],
                        }
                        for idx, s in enumerate(sections)
                        if isinstance(s, dict)
                    ]

                # Build hierarchical postings
                self.category_postings[category].append(doc_id)
                self.technology_postings[raw_tech].append(doc_id)
                self.domain_postings[domain].append(doc_id)

    def get_postings(self, term: str) -> list[Posting]:
        """Retrieve the postings list for a given term."""
        return self.postings.get(term, [])

    def document_frequency(self, term: str) -> int:
        """Return the number of documents containing the term (df)."""
        return len(self.postings.get(term, []))

    def idf(self, term: str) -> float:
        """
        Compute Inverse Document Frequency for a term.

        Uses the BM25 variant:
            IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        where N = total docs, df(t) = docs containing term t.
        """
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def get_zone_length(self, doc_id: str, zone: str) -> int:
        """Get the token count for a specific zone of a document."""
        return self.doc_zone_lengths.get(doc_id, {}).get(zone, 0)

    def get_total_doc_length(self, doc_id: str) -> int:
        """Get the total token count across all zones for a document."""
        return sum(self.doc_zone_lengths.get(doc_id, {}).values())

    def get_avg_total_doc_length(self) -> float:
        """Get the average total document length across all zones."""
        return sum(self.avg_zone_lengths.values())

    def get_docs_in_category(self, category: str) -> list[str]:
        """Return list of doc_ids belonging to a specific category."""
        return self.category_postings.get(category, [])

    def get_docs_in_technology(self, technology: str) -> list[str]:
        """Return list of doc_ids belonging to a specific technology."""
        return self.technology_postings.get(technology, [])

    def get_docs_in_domain(self, domain: str) -> list[str]:
        """Return list of doc_ids belonging to a specific top-level domain."""
        return self.domain_postings.get(domain, [])

    # ── Multi-Level Tiered & Section Retrieval Methods ──────────────────────

    def build_champion_lists(self, top_r: int = 50) -> None:
        """
        Constructs Tier-1 (Champion Lists) and Tier-2 postings for every term.
        Tier-1 contains postings where the term appears in high-value zones (title/heading)
        or has the highest total term frequency (top r).
        Tier-2 contains remaining postings.
        """
        self.champion_postings.clear()
        self.tier2_postings.clear()

        for term, plist in self.postings.items():
            high_value: list[Posting] = []
            standard: list[Posting] = []

            for p in plist:
                has_prominent_zone = (p.zone_tfs.get("title", 0) > 0) or (p.zone_tfs.get("heading", 0) > 0)
                if has_prominent_zone:
                    high_value.append(p)
                else:
                    standard.append(p)

            # Sort standard by total frequency descending
            standard.sort(key=lambda p: -sum(p.zone_tfs.values()))

            # Combine high_value + top standard up to top_r
            champ = high_value + standard[:max(0, top_r - len(high_value))]
            champ_set = {p.doc_id for p in champ}
            tier2 = [p for p in plist if p.doc_id not in champ_set]

            self.champion_postings[term] = champ
            self.tier2_postings[term] = tier2

    def get_champion_postings(self, term: str) -> list[Posting]:
        """Return Tier-1 Champion postings for a term."""
        if not self.champion_postings and self.postings:
            self.build_champion_lists()
        return self.champion_postings.get(term, [])

    def get_tier2_postings(self, term: str) -> list[Posting]:
        """Return Tier-2 postings for a term."""
        if not self.tier2_postings and self.postings:
            self.build_champion_lists()
        return self.tier2_postings.get(term, [])

    def score_sections(
        self,
        query_terms: list[str],
        top_k: int = 10,
        doc_id_filter: set[str] | list[str] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        """
        Multi-level section/passage retrieval: scores individual sections
        using BM25, optionally constrained to specific top retrieved documents.
        """
        if isinstance(doc_id_filter, list):
            doc_id_filter = set(doc_id_filter)

        sec_scores: dict[str, float] = defaultdict(float)
        avg_len = self.avg_section_length if self.avg_section_length > 0 else 50.0

        for term in query_terms:
            idf = self.idf(term)
            if idf <= 0.0:
                continue

            for sec_id, tf in self.section_postings.get(term, []):
                doc_id = sec_id.split("#")[0]
                if doc_id_filter is not None and doc_id not in doc_id_filter:
                    continue

                sec_len = self.section_lengths.get(sec_id, int(avg_len))
                len_norm = 1.0 - b + b * (sec_len / avg_len) if avg_len > 0 else 1.0
                term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                sec_scores[sec_id] += term_score

        if not sec_scores:
            return []

        ranked = sorted(sec_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for rank, (sec_id, score) in enumerate(ranked, start=1):
            doc_id = sec_id.split("#")[0]
            doc_meta = self.doc_metadata.get(doc_id, {})
            sec_meta = {}
            for s in self.doc_sections.get(doc_id, []):
                if s.get("sec_id") == sec_id:
                    sec_meta = s
                    break

            results.append({
                "rank": rank,
                "section_id": sec_id,
                "doc_id": doc_id,
                "score": round(score, 6),
                "heading": sec_meta.get("heading", ""),
                "snippet": sec_meta.get("snippet", ""),
                "document_title": doc_meta.get("title", ""),
                "url": doc_meta.get("url", ""),
                "technology": doc_meta.get("technology", ""),
                "category": doc_meta.get("category", ""),
            })

        return results

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Serialize the inverted index to disk as a pure inverted index.
        Also saves document catalog (lengths, averages, metadata, sections) to document_catalog.json.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Pure Inverted Index (term -> {df, idf, postings})
        index_data = {}
        for term, plist in self.postings.items():
            df = len(plist)
            idf = self.idf(term)
            index_data[term] = {
                "df": df,
                "idf": round(idf, 6),
                "postings": [
                    {
                        "doc_id": p.doc_id,
                        "tfs": p.zone_tfs,
                        "positions": p.positions,
                    }
                    for p in plist
                ],
            }

        with path.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False)

        # 2. Document Catalog (lengths, averages, metadata, multi-level hierarchy)
        catalog_path = path.parent / "document_catalog.json"
        catalog_data = {
            "total_docs": self.total_docs,
            "vocabulary_size": len(self.vocabulary),
            "avg_zone_lengths": self.avg_zone_lengths,
            "doc_zone_lengths": self.doc_zone_lengths,
            "doc_metadata": self.doc_metadata,
            "category_postings": dict(self.category_postings),
            "technology_postings": dict(self.technology_postings),
            "domain_postings": dict(self.domain_postings),
            "doc_sections": dict(self.doc_sections),
            "avg_section_length": self.avg_section_length,
        }
        with catalog_path.open("w", encoding="utf-8") as f:
            json.dump(catalog_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path, catalog_path: Path | None = None) -> "InvertedIndex":
        """Deserialize an inverted index and its document catalog from disk."""
        if catalog_path is None:
            catalog_path = path.parent / "document_catalog.json"

        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        idx = cls()

        # Check if legacy wrapped format or clean index format
        if "postings" in raw_data and "total_docs" in raw_data:
            idx.total_docs = raw_data["total_docs"]
            idx.avg_zone_lengths = raw_data["avg_zone_lengths"]
            idx.doc_zone_lengths = raw_data["doc_zone_lengths"]
            idx.doc_metadata = raw_data["doc_metadata"]
            postings_dict = raw_data["postings"]
        else:
            if catalog_path.exists():
                with catalog_path.open("r", encoding="utf-8") as f:
                    cat = json.load(f)
                idx.total_docs = cat["total_docs"]
                idx.avg_zone_lengths = cat["avg_zone_lengths"]
                idx.doc_zone_lengths = cat["doc_zone_lengths"]
                idx.doc_metadata = cat["doc_metadata"]
                if "category_postings" in cat:
                    idx.category_postings = defaultdict(list, cat["category_postings"])
                if "technology_postings" in cat:
                    idx.technology_postings = defaultdict(list, cat["technology_postings"])
                if "domain_postings" in cat:
                    idx.domain_postings = defaultdict(list, cat["domain_postings"])
                if "doc_sections" in cat:
                    idx.doc_sections = defaultdict(list, cat["doc_sections"])
                if "avg_section_length" in cat:
                    idx.avg_section_length = cat["avg_section_length"]
            postings_dict = raw_data

        # Fallback rebuild of hierarchical postings if missing
        if not idx.category_postings and idx.doc_metadata:
            for did, m in idx.doc_metadata.items():
                cat = m.get("category", "")
                tech = m.get("technology", "")
                dom = m.get("domain", "")
                if cat: idx.category_postings[cat].append(did)
                if tech: idx.technology_postings[tech].append(did)
                if dom: idx.domain_postings[dom].append(did)

        idx.vocabulary = set()
        for term, val in postings_dict.items():
            idx.vocabulary.add(term)
            plist_raw = val["postings"] if isinstance(val, dict) and "postings" in val else val
            idx.postings[term] = []
            for p in plist_raw:
                tfs = p.get("tfs", p.get("zone_tfs", {}))
                pos = p.get("positions", {})
                if isinstance(pos, list):
                    # convert legacy list of pairs [("zone", offset)]
                    pos_dict: dict[str, list[int]] = defaultdict(list)
                    for item in pos:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            pos_dict[item[0]].append(item[1])
                    pos = dict(pos_dict)
                idx.postings[term].append(
                    Posting(
                        doc_id=p["doc_id"],
                        zone_tfs=tfs,
                        positions=pos,
                    )
                )

        return idx

    def get_index_stats(self) -> dict[str, Any]:
        """Return summary statistics about the index."""
        total_postings = sum(len(pl) for pl in self.postings.values())
        total_positions = sum(
            sum(len(offsets) for offsets in p.positions.values()) if isinstance(p.positions, dict) else len(p.positions)
            for pl in self.postings.values()
            for p in pl
        )
        return {
            "total_documents": self.total_docs,
            "vocabulary_size": len(self.vocabulary),
            "total_postings_entries": total_postings,
            "total_position_entries": total_positions,
            "avg_postings_per_term": round(total_postings / len(self.vocabulary), 2) if self.vocabulary else 0,
            "avg_zone_lengths": {z: round(v, 1) for z, v in self.avg_zone_lengths.items()},
        }
