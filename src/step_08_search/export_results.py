"""Export fixed-query retrieval snapshots and measured resource usage."""

import csv
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time
import numpy as np
from src.common.artifacts import digest_file, write_json, MODEL_ID, MODEL_REVISION
from src.common.zones import ZONES

COLUMNS = [
    "engine",
    "variant",
    "query_id",
    "query_text",
    "rank",
    "doc_id",
    "title",
    "url",
    "technology",
    "categories",
    "score",
]


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def export_results(
    engine, query_path, variants_path, directory, startup_seconds, repeats=3
):
    queries = [
        json.loads(line)
        for line in Path(query_path).read_text().splitlines()
        if line.strip()
    ]
    variants = json.loads(Path(variants_path).read_text())
    if (
        len(queries) != 20
        or len({q["query_id"] for q in queries}) != 20
        or len({q["query_text"] for q in queries}) != 20
    ):
        raise ValueError("Export requires exactly 20 distinct queries")
    if len({v["id"] for v in variants}) != len(variants):
        raise ValueError("Duplicate variant ID")
    for variant in variants:
        if variant["mode"] not in {"dense", "zones", "routed-centroid"}:
            raise ValueError("Unsupported variant")
        if variant["weights"] is not None:
            weights = variant["weights"]
            if (
                set(weights) != set(ZONES)
                or any(not np.isfinite(w) or w < 0 for w in weights.values())
                or not np.isclose(sum(weights.values()), 1)
            ):
                raise ValueError("Variant weights must be nonnegative and sum to one")
        if not all(c.isalnum() or c in "_-" for c in variant["id"]):
            raise ValueError("Invalid variant ID")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cuda = engine.device == "cuda"
    sync = lambda: engine.torch.cuda.synchronize() if cuda else None
    engine.warmup()
    source = json.loads((engine.root / "step_01_extraction/summary.json").read_text())
    original = engine.weights.copy()
    manifests = []
    timings = []
    try:
        for variant in variants:
            if variant["weights"] is not None:
                engine.weights = np.array(
                    [variant["weights"][z] for z in ZONES], dtype=np.float32
                )
            rows = []
            variant_times = []
            for query in queries:
                samples = []
                for repetition in range(repeats):
                    engine.query_cache.clear()
                    if cuda:
                        engine.torch.cuda.reset_peak_memory_stats()
                    before = rss_bytes()
                    sync()
                    started = time.perf_counter()
                    cpu = time.process_time()
                    engine._query(query["query_text"])
                    sync()
                    encoded = time.perf_counter()
                    ranking = engine.rank(query["query_text"], variant["mode"])
                    sync()
                    finished = time.perf_counter()
                    sample = {
                        "engine": "neural",
                        "variant": variant["id"],
                        "query_id": query["query_id"],
                        "repetition": repetition + 1,
                        "retrieval_ms": (finished - started) * 1000,
                        "query_embedding_ms": (encoded - started) * 1000,
                        "ranking_ms": (finished - encoded) * 1000,
                        "cpu_ms": (time.process_time() - cpu) * 1000,
                        "rss_before_bytes": before,
                        "rss_after_bytes": rss_bytes(),
                        "process_high_water_rss_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                        "cuda_peak_allocated_bytes": engine.torch.cuda.max_memory_allocated()
                        if cuda
                        else 0,
                        "cuda_peak_reserved_bytes": engine.torch.cuda.max_memory_reserved()
                        if cuda
                        else 0,
                        "scanned_vectors": ranking["scanned_vectors"],
                        "candidate_documents": ranking["candidate_documents"],
                    }
                    samples.append(sample)
                    timings.append(sample)
                variant_times.extend(s["retrieval_ms"] for s in samples)
                order = ranking["order"][:10]
                if len(order) != 10:
                    raise RuntimeError("A query returned fewer than 10 documents")
                for rank, index in enumerate(order, 1):
                    index = int(index)
                    doc = engine.documents[index]
                    categories = [
                        engine.classifier.model["class_names"][
                            engine.classifier.model["classes"][int(c)]
                        ]
                        for c in engine.classifier.document_classes[index]
                        if c >= 0
                    ]
                    rows.append(
                        dict(
                            engine="neural",
                            variant=variant["id"],
                            query_id=query["query_id"],
                            query_text=query["query_text"],
                            rank=rank,
                            doc_id=doc["doc_id"],
                            title=doc["title"],
                            url=doc["url"],
                            technology=doc["technology"],
                            categories=json.dumps(categories),
                            score=float(ranking["scores"][index]),
                        )
                    )
            write_csv(directory / f"{variant['id']}.csv", rows, COLUMNS)
            manifests.append(
                {
                    "engine": "neural",
                    "variant": variant["id"],
                    "mode": variant["mode"],
                    "weights": variant["weights"],
                    "route_classes": engine.config["route_classes"]
                    if variant["mode"] == "routed-centroid"
                    else None,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "dimensions": engine.encoder.dimension,
                    "index_fingerprint": engine.summary["fingerprint"],
                    "corpus_manifest_sha256": source["dataset_manifest_sha256"],
                    "corpus_documents": len(engine.documents),
                    "query_count": 20,
                    "results_per_query": 10,
                    "repetitions": repeats,
                    "queries_sha256": digest_file(query_path),
                    "variants_sha256": digest_file(variants_path),
                    "device": engine.device,
                    "python": platform.python_version(),
                    "torch": engine.torch.__version__,
                    "startup_seconds": startup_seconds,
                    "retrieval_median_ms": statistics.median(variant_times),
                    "retrieval_p95_ms": float(np.percentile(variant_times, 95)),
                    "vector_bytes": sum(m.nbytes for m in engine.matrices.values()),
                    "vector_and_mapping_file_bytes": sum(
                        (engine.root / f"step_03_embeddings/{z}/{f}").stat().st_size
                        for z in ZONES
                        for f in ("vectors.npy", "doc_indices.npy")
                    ),
                    "timing_scope": "Warm model; uncached query encoding plus ranking and sorting; excludes metadata serialization and startup",
                    "memory_scope": "RSS before/after plus process lifetime high-water mark; CUDA peaks include resident model and index",
                    "query_cache": False,
                    "ranking_cache": False,
                }
            )
            print(
                f"Exported {variant['id']}: 20 queries, 200 results, median {statistics.median(variant_times):.1f} ms",
                flush=True,
            )
            write_csv(directory / "timings.csv", timings, list(timings[0]))
            write_json(directory / "manifest.json", manifests)
    finally:
        engine.weights = original
        engine.query_cache.clear()
