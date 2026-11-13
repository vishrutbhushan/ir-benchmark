"""Complete zone-index, reference cosine, retrieval and cache verification."""

import csv
from pathlib import Path
import time

import numpy as np

from src.common.artifacts import digest_file, report
from src.common.zones import ZONES
from src.verification.integrity import audit_zone_coverage, audit_vector_mapping
from src.retrieval.scoring import aggregate_documents
from src.retrieval.step_08_search import MODES
from src.common.config import load_config
from src.common.paths import REPOSITORY_ROOT


def verify(engine, root):
    root = Path(root)
    started = time.perf_counter()
    from src.verification.learning_audit import audit_learning_data

    if engine.config["use_learned_zone_weights"]:
        audit_learning_data(root)
    learning_data_audit = "passed" if engine.config["use_learned_zone_weights"] else "disabled"
    assignments = engine.clusters.labels
    if assignments.shape != (len(engine.documents),) or np.any(
        assignments < 0
    ) or np.any(assignments >= engine.clusters.model["clusters"]):
        raise RuntimeError("Invalid full-corpus cluster assignments")
    for c in range(engine.clusters.model["clusters"]):
        allowed = np.any(engine.clusters.memberships == c, axis=1)
        for z in ZONES:
            expected = np.flatnonzero(allowed[engine.mappings[z]])
            if not np.array_equal(expected, engine.routing_rows[f"{c}_{z}"]):
                raise RuntimeError(
                    "Cluster row lists do not match document assignments"
                )
    with np.load(root / "step_05_clustering/zone_vectors.npz") as saved:
        for column in range(2):
            vectors=saved["vectors"][:,column].astype(np.float32)
            norms=np.linalg.norm(vectors[saved["present"][:,column]],axis=1)
            if np.any(np.abs(norms-1)>0.002):
                raise RuntimeError("Clustering zone vectors are not normalized")
    coverage = audit_zone_coverage(engine.document_db, engine.stores)
    q = engine._query("Python asynchronous task cancellation")
    actual, _, _ = engine.zone_scores(q)
    hashes = {}
    norms = {}
    errors = {}
    for column, zone in enumerate(ZONES):
        matrix, mapping = engine.matrices[zone], engine.mappings[zone]
        if audit_vector_mapping(engine.stores[zone], mapping) != len(matrix):
            raise RuntimeError("Vector mapping size mismatch")
        scores = np.empty(len(matrix), dtype=np.float32)
        low, high = 2.0, 0.0
        for start in range(0, len(matrix), 8192):
            block = np.asarray(matrix[start : start + 8192], dtype=np.float32)
            length = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or np.any(np.abs(length - 1) > 0.002):
                raise RuntimeError("Invalid persisted vectors")
            low, high = min(low, float(length.min())), max(high, float(length.max()))
            scores[start : start + len(block)] = (block @ q) / (
                length * np.linalg.norm(q)
            )
        reference, _ = aggregate_documents(scores, mapping, len(engine.documents))
        present = np.isfinite(reference)
        if not np.array_equal(present, np.isfinite(actual[:, column])):
            raise RuntimeError("Missing-zone score mismatch")
        error = float(np.max(np.abs(reference[present] - actual[present, column])))
        if error > 1e-4:
            raise RuntimeError("GPU/NumPy cosine disagreement")
        errors[zone] = error
        norms[zone] = [low, high]
        hashes[zone] = digest_file(root / f"step_03_embeddings/{zone}/vectors.npy")
    settings = load_config()
    development_path = Path(settings["weight_queries"])
    if not development_path.is_absolute():
        development_path = REPOSITORY_ROOT / development_path
    with development_path.open(newline="", encoding="utf-8-sig") as stream:
        by_query = {
            r["query_id"]: {
                "query_id": r["query_id"],
                "query_text": r["query_text"],
            }
            for r in csv.DictReader(stream)
        }
    queries = list(by_query.values())
    timings = {}
    counts = {}
    for mode in MODES:
        elapsed = []
        for query in queries:
            result = engine.search(query["query_text"], mode=mode, use_cache=False)
            ids = [r["doc_id"] for r in result["results"]]
            expected = min(
                engine.config["page_size"],
                engine.config["result_limit"],
                len(engine.documents),
            )
            if len(ids) != expected or len(ids) != len(set(ids)):
                raise RuntimeError("Invalid result depth/IDs")
            scores = [r["score"] for r in result["results"]]
            if scores != sorted(scores, reverse=True):
                raise RuntimeError("Rank order mismatch")
            elapsed.append(result["elapsed_ms"])
        timings[mode + "_p50_ms"] = float(np.median(elapsed))
        timings[mode + "_p95_ms"] = float(np.percentile(elapsed, 95))
        counts[mode] = len(queries)
        print(f"Verified {mode}: {len(queries)} queries.", flush=True)
    query = queries[0]["query_text"]
    size = engine.config["page_size"]
    first = engine.search(query)
    paged = first["total_results"] > size
    second = engine.search(query, offset=size if paged else 0)
    if not second["cache_hit"]:
        raise RuntimeError("Ranking cache mismatch")
    if paged and (
        second["results"][0]["rank"] != size + 1
        or {r["doc_id"] for r in first["results"]}
        & {r["doc_id"] for r in second["results"]}
    ):
        raise RuntimeError("Pagination mismatch")
    stats = {
        "index_fingerprint": engine.summary["fingerprint"],
        "documents": len(engine.documents),
        "all_cluster_row_lists_checked": True,
        "all_document_assignments_checked": True,
        "windows": sum(len(m) for m in engine.matrices.values()),
        "coverage": coverage,
        "vectors_sha256": hashes,
        "norm_ranges": norms,
        "cosine_reference_max_error": errors,
        "query_runs": counts,
        "all_vector_document_mappings_checked": True,
        "cached_pagination_checked": True,
        "learning_data_audit": learning_data_audit,
        "development_csv_sha256": digest_file(development_path),
        "query_scope": "Development queries for engineering checks; benchmark queries belong to evaluation branch",
        "seconds": round(time.perf_counter() - started, 2),
        "latencies": timings,
    }
    report(
        root / "step_07_verification",
        "Complete neural verification",
        stats,
        {
            "latencies_ms": timings,
        },
    )
    return stats
