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


# Zone names used throughout the index
ZONES = ("title", "heading", "body", "code")


class Posting(NamedTuple):
    """A single posting entry in the inverted index."""
    doc_id: str
    zone_tfs: dict[str, int]       # {"title": 2, "heading": 0, "body": 5, "code": 1}
    positions: list[tuple[str, int]]  # [("title", 0), ("body", 12), ("body", 45), ...]


class InvertedIndex:
    """
    Hand-coded inverted index with per-zone term frequencies and positional
    information.

    Attributes:
        postings: Mapping from term → list of Posting entries.
        doc_zone_lengths: Mapping from doc_id → {zone: token_count}.
        avg_zone_lengths: Corpus-wide average token count per zone.
        total_docs: Total number of documents indexed.
        doc_metadata: Mapping from doc_id → metadata dict (category, technology, url, etc.).
        vocabulary: Set of all unique terms in the index.
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
                        self.vocabulary.add(token)

                        if doc_id not in temp_postings[token]:
                            temp_postings[token][doc_id] = {
                                "zone_tfs": {z: 0 for z in ZONES},
                                "positions": [],
                            }

                        temp_postings[token][doc_id]["zone_tfs"][zone] += 1
                        temp_postings[token][doc_id]["positions"].append((zone, position))

                self.doc_zone_lengths[doc_id] = zone_lengths

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

        # Step 4: Convert temp_postings to final Posting objects
        logger.info("Finalizing postings lists...")
        for term, doc_entries in temp_postings.items():
            posting_list: list[Posting] = []
            for did, data in doc_entries.items():
                posting_list.append(
                    Posting(
                        doc_id=did,
                        zone_tfs=data["zone_tfs"],
                        positions=data["positions"],
                    )
                )
            # Sort postings by doc_id for deterministic output
            posting_list.sort(key=lambda p: p.doc_id)
            self.postings[term] = posting_list

        total_time = time.time() - start_time
        logger.info(
            f"Inverted index built in {total_time:.1f}s. "
            f"Documents: {self.total_docs}, "
            f"Vocabulary: {len(self.vocabulary):,} terms, "
            f"Total postings entries: {sum(len(pl) for pl in self.postings.values()):,}"
        )
        logger.info(
            f"Avg zone lengths: "
            + ", ".join(f"{z}={self.avg_zone_lengths[z]:.1f}" for z in ZONES)
        )

    def _load_metadata(self, structured_path: Path) -> None:
        """Load document metadata from structured records."""
        with structured_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc_id = record["doc_id"]
                self.doc_metadata[doc_id] = {
                    "url": record.get("url", ""),
                    "title": record.get("title", ""),
                    "technology": record.get("metadata", {}).get("technology", ""),
                    "category": record.get("metadata", {}).get("category", ""),
                    "dates": record.get("dates", []),
                }

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

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize the inverted index to disk as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_docs": self.total_docs,
            "vocabulary_size": len(self.vocabulary),
            "avg_zone_lengths": self.avg_zone_lengths,
            "doc_zone_lengths": self.doc_zone_lengths,
            "doc_metadata": self.doc_metadata,
            "postings": {
                term: [
                    {
                        "doc_id": p.doc_id,
                        "zone_tfs": p.zone_tfs,
                        "positions": p.positions,
                    }
                    for p in plist
                ]
                for term, plist in self.postings.items()
            },
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        """Deserialize an inverted index from disk."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls()
        idx.total_docs = data["total_docs"]
        idx.avg_zone_lengths = data["avg_zone_lengths"]
        idx.doc_zone_lengths = data["doc_zone_lengths"]
        idx.doc_metadata = data["doc_metadata"]
        idx.vocabulary = set()

        for term, plist in data["postings"].items():
            idx.vocabulary.add(term)
            idx.postings[term] = [
                Posting(
                    doc_id=p["doc_id"],
                    zone_tfs=p["zone_tfs"],
                    positions=[tuple(pos) for pos in p["positions"]],
                )
                for p in plist
            ]

        return idx

    def get_index_stats(self) -> dict[str, Any]:
        """Return summary statistics about the index."""
        total_postings = sum(len(pl) for pl in self.postings.values())
        total_positions = sum(
            len(p.positions)
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
