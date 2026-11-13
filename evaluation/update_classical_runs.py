#!/usr/bin/env python3
"""Export classical pipeline rankings and fuse BM25F with routed neural results."""

import json
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from config import load_pipeline_config, resolve_path
from evaluation.src.common import COLUMNS, read_csv, write_csv


def export_classical(run_dir, destination, queries, limit):
    expected = {query["query_id"]: query["query_text"] for query in queries}
    exports = {}
    for variant in ("bm25", "bm25f"):
        source = json.loads((run_dir / f"{variant}_results.json").read_text(encoding="utf-8"))
        if len(source) != len(expected) or {row["query_id"] for row in source} != set(expected):
            raise ValueError(f"{variant} pipeline results do not match the configured queries")
        rows = []
        for query in source:
            qid = query["query_id"]
            results = sorted(query["results"], key=lambda row: int(row["rank"]))[:limit]
            if query["query_text"] != expected[qid] or len(results) != limit:
                raise ValueError(f"Invalid {variant} result text/depth for {qid}")
            if len({row["doc_id"] for row in results}) != limit:
                raise ValueError(f"Duplicate {variant} results for {qid}")
            for rank, result in enumerate(results, 1):
                rows.append({
                    "engine": "classical", "variant": variant,
                    "query_id": qid, "query_text": expected[qid], "rank": rank,
                    "doc_id": result["doc_id"], "title": result.get("title", ""),
                    "url": result.get("url", ""), "technology": result.get("technology", ""),
                    "categories": json.dumps([result["category"]] if result.get("category") else []),
                    "score": result["score"],
                })
        exports[variant] = rows
    for variant, rows in exports.items():
        write_csv(destination / f"{variant}.csv", rows, COLUMNS)


def read_routed(path, limit):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    for query_id, query_rows in grouped.items():
        grouped[query_id] = sorted(query_rows, key=lambda row: int(row["rank"]))[:limit]
    return grouped


def read_classical_csv(path):
    rows = read_csv(path)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(
            {**row, "rank": int(row["rank"])}
        )
    output = []
    for query_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["rank"])
        limit = len(ordered)
        if [row["rank"] for row in ordered] != list(range(1, limit + 1)):
            raise ValueError(f"Invalid BM25F ranks for {query_id}")
        if len({row["doc_id"] for row in ordered}) != limit:
            raise ValueError(f"Duplicate BM25F document for {query_id}")
        output.append({"query_id": query_id, "results": ordered})
    return output


def build_rrf(bm25f_queries, routed, config):
    settings = config["evaluation"]["rrf"]
    bm25f_depth = settings["bm25f_depth"]
    routed_depth = settings["routed_depth"]
    rows = []
    for query in bm25f_queries:
        query_id = query["query_id"]
        scores = {}
        for rank, result in enumerate(query["results"][:bm25f_depth], 1):
            doc_id = result["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + settings["bm25f_weight"] / (settings["k"] + rank)
        for rank, result in enumerate(routed.get(query_id, [])[:routed_depth], 1):
            doc_id = result["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + settings["routed_weight"] / (settings["k"] + rank)
        for rank, (doc_id, score) in enumerate(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)[: config["evaluation"]["results_per_query"]],
            1,
        ):
            rows.append(
                {
                    "engine": "hybrid",
                    "variant": "rrf",
                    "query_id": query_id,
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": f"{score:.6f}",
                }
            )
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT.parent / "config/pipeline.json")
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    results_dir = resolve_path(config["paths"]["evaluation_results_dir"])
    classical_dir = results_dir / "classical"
    out_dir = classical_dir
    limit = config["evaluation"]["results_per_query"]
    run_dir = resolve_path(config["paths"]["classical_output_dir"]) / "stage_03_retrieval"
    queries = [
        json.loads(line)
        for line in resolve_path(config["paths"]["queries"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    export_classical(run_dir, classical_dir, queries, limit)
    bm25f_path = classical_dir / "bm25f.csv"
    bm25f_queries = read_classical_csv(bm25f_path)
    expected_queries = {
        json.loads(line)["query_id"]
        for line in resolve_path(config["paths"]["queries"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if {q["query_id"] for q in bm25f_queries} != expected_queries:
        raise ValueError("BM25F result CSV does not match the configured query suite")
    if any(len(q["results"]) < config["evaluation"]["rrf"]["bm25f_depth"] for q in bm25f_queries):
        raise ValueError("BM25F result CSV is shallower than the configured RRF depth")
    routed = read_routed(results_dir / "neural/routed.csv", config["evaluation"]["rrf"]["routed_depth"])
    if set(routed) != expected_queries or any(len(rows) < config["evaluation"]["rrf"]["routed_depth"] for rows in routed.values()):
        raise ValueError("Routed neural result CSV does not match the configured RRF input depth")
    rrf_rows = build_rrf(bm25f_queries, routed, config)
    write_csv(
        results_dir / "hybrid/rrf.csv",
        rrf_rows,
        ["engine", "variant", "query_id", "rank", "doc_id", "score"],
    )
    print("Updated results/hybrid/rrf.csv from the current BM25F and routed snapshots")

    summary = json.loads((run_dir / "phase3_summary.json").read_text(encoding="utf-8"))
    corpus_summary = json.loads(
        (
            resolve_path(config["paths"]["filter_output_dir"])
            / "stage_04_filtering/summary.json"
        ).read_text(encoding="utf-8")
    )
    manifest = [
        {
            "engine": "classical",
            "variant": variant,
            "corpus_documents": summary["index_statistics"]["total_documents"],
            "corpus_manifest_sha256": corpus_summary["dataset_manifest_sha256"],
            "parameters_from_config": (
                {
                    "k1": config["classical"]["k1"],
                    "b": config["classical"]["b"],
                    "top_k": config["classical"]["top_k"],
                }
                if variant == "bm25"
                else {
                    "k1": config["classical"]["k1"],
                    "top_k": config["classical"]["top_k"],
                    "hierarchical_boost": config["classical"]["hierarchical_boost"],
                    "zone_weights": config["classical"]["zone_weights"],
                    "zone_b": config["classical"]["zone_b"],
                    "routing_mode": config["classical"]["routing_mode"],
                }
            ),
            "query_count": len(expected_queries),
            "results_per_query": limit,
        }
        for variant in ("bm25", "bm25f")
    ]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Updated results/classical/manifest.json")


if __name__ == "__main__":
    main()
