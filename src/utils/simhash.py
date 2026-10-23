from __future__ import annotations

import hashlib
import re
from typing import Iterable


def get_tokens(text: str) -> list[str]:
    """Extracts lowercase alphanumeric tokens for hashing."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


def get_shingles(tokens: list[str], k: int = 3) -> list[str]:
    """Generates k-shingles (word n-grams) from tokens."""
    if len(tokens) < k:
        return ["_".join(tokens)] if tokens else []
    return ["_".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def _hash64(feature: str) -> int:
    """Computes a 64-bit hash of a string feature."""
    return int(hashlib.md5(feature.encode("utf-8")).hexdigest()[:16], 16)


def compute_simhash(features: Iterable[str], bit_size: int = 64) -> int:
    """Computes bit_size SimHash fingerprint for an iterable of feature strings."""
    v = [0] * bit_size
    for feature in features:
        h = _hash64(feature)
        for i in range(bit_size):
            bitmask = 1 << i
            if h & bitmask:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(bit_size):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates the Hamming distance (number of differing bits) between two hashes."""
    x = hash1 ^ hash2
    return bin(x).count("1")


def is_near_duplicate(hash1: int, hash2: int, max_distance: int = 3) -> bool:
    """Checks if two SimHash fingerprints represent near-duplicate documents."""
    return hamming_distance(hash1, hash2) <= max_distance
