"""
Hand-coded 64-bit SimHash & Content Deduplicator for Technical Web Corpus.

Detects and filters:
1. Exact content duplicates (SHA-256 over normalized title + body_text + code_text)
2. Near-duplicate pages across documentation versions (e.g., v1.13 vs v1.15 of the
   same page) using 64-bit SimHash over 3-word shingles (Hamming distance <= 3)
   and normalized title + body prefix fingerprints.
"""

from __future__ import annotations

import collections
import hashlib
import re
from typing import Any, Iterable

_VERSION_NOISE_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_\-\.]{2,}\b")


def get_tokens(text: str, max_chars: int = 8000) -> list[str]:
    """Extracts lowercase alphanumeric tokens for SimHash shingling."""
    return _TOKEN_RE.findall(text[:max_chars].lower())


def get_shingles(tokens: list[str], k: int = 3) -> list[str]:
    """Generates k-shingles (word n-grams) from tokens."""
    if len(tokens) < k:
        return ["_".join(tokens)] if tokens else []
    # Cap at 800 shingles per document for fast 64-bit SimHash computation
    limit = min(len(tokens) - k + 1, 800)
    return ["_".join(tokens[i : i + k]) for i in range(limit)]


def _hash64(feature: str) -> int:
    """Computes a 64-bit hash of a string feature."""
    return int.from_bytes(hashlib.md5(feature.encode("utf-8")).digest()[:8], "big")


def compute_simhash(features: Iterable[str], bit_size: int = 64) -> int:
    """Computes a 64-bit SimHash fingerprint for an iterable of feature strings."""
    v = [0] * bit_size
    for feature in features:
        h = _hash64(feature)
        for i in range(bit_size):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(bit_size):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates the Hamming distance (number of differing bits) between two 64-bit hashes."""
    return (hash1 ^ hash2).bit_count()


def is_near_duplicate(hash1: int, hash2: int, max_distance: int = 3) -> bool:
    """Checks if two SimHash fingerprints represent near-duplicate documents."""
    return hamming_distance(hash1, hash2) <= max_distance


def normalize_for_dedup(text: str, limit: int = 600) -> str:
    """Normalizes text by stripping version strings and collapsing whitespace."""
    cleaned = _VERSION_NOISE_RE.sub("", text[:limit].lower())
    return _WS_RE.sub(" ", cleaned).strip()


class CorpusDeduplicator:
    """
    Stateful deduplicator for Phase 1 structured extraction and corpus pruning.

    Tracks:
    - Exact SHA-256 content hashes (title + body_text + code_text)
    - Version-normalized (technology, title, body_prefix) keys
    - 64-bit SimHash buckets partitioned by 16-bit blocks (pigeonhole principle
      for Hamming distance <= 3) within the same technology
    """

    def __init__(self, max_hamming_distance: int = 3) -> None:
        self.max_hamming_distance = max_hamming_distance
        self.seen_exact_hashes: dict[str, str] = {}
        self.seen_norm_keys: dict[tuple[str, str, str], str] = {}
        # (tech, block_idx, 16bit_val) -> list of (doc_id, simhash)
        self.simhash_buckets: dict[tuple[str, int, int], list[tuple[str, int]]] = (
            collections.defaultdict(list)
        )
        self.exact_duplicates_count: int = 0
        self.near_duplicates_count: int = 0
        self.duplicate_to_canonical: dict[str, str] = {}

    def check_and_register(self, record: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Checks if `record` is an exact or near-duplicate of an already-registered document.

        Returns:
            (is_duplicate, canonical_doc_id_if_duplicate)
        """
        doc_id = record.get("doc_id", "")
        title = (record.get("title") or "").strip()
        body = (record.get("body_text") or "").strip()
        code = (record.get("code_text") or "").strip()
        tech = (record.get("metadata", {}).get("technology") or "").strip().lower()

        # 1. Exact content hash check
        exact_payload = f"{title.lower()}||{body}||{code}".encode("utf-8", errors="ignore")
        exact_hash = hashlib.sha256(exact_payload).hexdigest()
        if exact_hash in self.seen_exact_hashes:
            canonical = self.seen_exact_hashes[exact_hash]
            self.exact_duplicates_count += 1
            self.duplicate_to_canonical[doc_id] = canonical
            return True, canonical

        # 2. Version-normalized title + body prefix check
        norm_title = normalize_for_dedup(title, limit=200)
        norm_body_prefix = normalize_for_dedup(body, limit=400)
        norm_key = (tech, norm_title, norm_body_prefix)
        if norm_body_prefix and norm_key in self.seen_norm_keys:
            canonical = self.seen_norm_keys[norm_key]
            self.near_duplicates_count += 1
            self.duplicate_to_canonical[doc_id] = canonical
            return True, canonical

        # 3. 64-bit SimHash near-duplicate check within the same technology & similar title
        tokens = get_tokens(body, max_chars=4000)
        shingles = get_shingles(tokens, k=3)
        simhash_val: int | None = None
        if len(shingles) >= 10:
            simhash_val = compute_simhash(shingles)
            checked_candidates: set[str] = set()
            for block_idx in range(4):
                block_val = (simhash_val >> (block_idx * 16)) & 0xFFFF
                for cand_id, cand_hash in self.simhash_buckets[(tech, block_idx, block_val)]:
                    if cand_id in checked_candidates:
                        continue
                    checked_candidates.add(cand_id)
                    if is_near_duplicate(simhash_val, cand_hash, self.max_hamming_distance):
                        self.near_duplicates_count += 1
                        self.duplicate_to_canonical[doc_id] = cand_id
                        return True, cand_id

        # Register as canonical document
        self.seen_exact_hashes[exact_hash] = doc_id
        if norm_body_prefix:
            self.seen_norm_keys[norm_key] = doc_id
        if simhash_val is not None:
            for block_idx in range(4):
                block_val = (simhash_val >> (block_idx * 16)) & 0xFFFF
                self.simhash_buckets[(tech, block_idx, block_val)].append((doc_id, simhash_val))

        return False, None

    def get_summary(self) -> dict[str, int]:
        """Returns counts of exact and near duplicates detected."""
        return {
            "exact_duplicates_removed": self.exact_duplicates_count,
            "near_duplicates_removed": self.near_duplicates_count,
            "total_duplicates_removed": self.exact_duplicates_count + self.near_duplicates_count,
            "unique_documents_retained": len(self.seen_exact_hashes),
        }

