from __future__ import annotations

import collections
from typing import Any

from src.utils.simhash import (
    compute_simhash,
    get_shingles,
    get_tokens,
    hamming_distance,
    is_near_duplicate,
)


def find_near_duplicate_clusters(
    documents: list[dict[str, Any]],
    max_hamming_distance: int = 3,
) -> dict[str, Any]:
    """
    Computes SimHash for each document's text and identifies near-duplicate clusters.
    Returns cluster mappings and statistics.
    """
    doc_hashes: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc.get("body_text", "")
        tokens = get_tokens(text)
        shingles = get_shingles(tokens, k=3)
        if not shingles:
            continue
        simhash_val = compute_simhash(shingles)
        doc_hashes.append({
            "doc_id": doc_id,
            "technology": doc.get("technology", "Unknown"),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "simhash": simhash_val,
            "char_length": len(text),
        })

    # Cluster documents
    clusters: list[list[dict[str, Any]]] = []
    visited: set[str] = set()

    # Pre-index by 16-bit blocks for fast candidate retrieval
    # SimHash 64-bit split into 4 16-bit chunks (pigeonhole principle for max distance <= 3)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for doc in doc_hashes:
        h = doc["simhash"]
        for block_idx in range(4):
            block_val = (h >> (block_idx * 16)) & 0xFFFF
            buckets[(block_idx, block_val)].append(doc)

    for doc in doc_hashes:
        doc_id = doc["doc_id"]
        if doc_id in visited:
            continue

        cluster = [doc]
        visited.add(doc_id)
        candidates = set()

        h = doc["simhash"]
        for block_idx in range(4):
            block_val = (h >> (block_idx * 16)) & 0xFFFF
            for candidate in buckets[(block_idx, block_val)]:
                cand_id = candidate["doc_id"]
                if cand_id != doc_id and cand_id not in visited:
                    candidates.add(cand_id)

        # Map candidate IDs back to doc objects
        cand_map = {d["doc_id"]: d for d in doc_hashes if d["doc_id"] in candidates}
        for cand_id, candidate in cand_map.items():
            if cand_id in visited:
                continue
            if is_near_duplicate(doc["simhash"], candidate["simhash"], max_hamming_distance):
                cluster.append(candidate)
                visited.add(cand_id)

        if len(cluster) > 1:
            clusters.append(cluster)

    cluster_report = {
        "total_documents_analyzed": len(doc_hashes),
        "total_clusters": len(clusters),
        "total_duplicate_documents": sum(len(c) for c in clusters),
        "clusters": [
            {
                "cluster_id": i + 1,
                "representative_doc_id": c[0]["doc_id"],
                "representative_title": c[0]["title"],
                "cluster_size": len(c),
                "member_doc_ids": [d["doc_id"] for d in c],
                "member_urls": [d["url"] for d in c],
            }
            for i, c in enumerate(clusters)
        ],
    }

    return cluster_report
