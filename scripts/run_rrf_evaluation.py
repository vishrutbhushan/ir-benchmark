#!/usr/bin/env python3
"""
Reciprocal Rank Fusion (RRF) Hybrid Evaluation.

Combines BM25F (lexical) and Dense (neural) ranking outputs across the 15
benchmark queries to compute hybrid rankings and effectiveness metrics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_rrf(
    bm25f_path: Path,
    dense_path: Path,
    judgments_path: Path,
    output_dir: Path,
    k: int = 20,
    w_dense: float = 0.7,
    w_bm25f: float = 0.3,
) -> dict[str, Any]:
    bm25f_rows = read_csv(bm25f_path)
    dense_rows = read_csv(dense_path)
    judgments = read_csv(judgments_path)

    labels = {(r["query_id"], r["doc_id"]): int(r.get("lenient", r.get("grade", 0))) for r in judgments}
    pool_docs: dict[str, set[str]] = {}
    for r in judgments:
        pool_docs.setdefault(r["query_id"], set()).add(r["doc_id"])

    queries = sorted(list(set(r["query_id"] for r in bm25f_rows)))
    output_dir.mkdir(parents=True, exist_ok=True)
    rrf_csv_path = output_dir / "rrf.csv"

    rrf_rows = []
    per_query_metrics = []
    p1_list, p3_list, p5_list, p10_list = [], [], [], []
    ndcg1_list, ndcg3_list, ndcg5_list, ndcg10_list = [], [], [], []
    mrr_list = []

    for qid in queries:
        b_df = [r for r in bm25f_rows if r["query_id"] == qid]
        d_df = [r for r in dense_rows if r["query_id"] == qid]

        b_ranks = {r["doc_id"]: int(r["rank"]) for r in b_df}
        d_ranks = {r["doc_id"]: int(r["rank"]) for r in d_df}
        all_docs = set(b_ranks.keys()) | set(d_ranks.keys())

        scores = {}
        for doc in all_docs:
            sc = 0.0
            if doc in d_ranks:
                sc += w_dense / (k + d_ranks[doc])
            if doc in b_ranks:
                sc += w_bm25f / (k + b_ranks[doc])
            scores[doc] = sc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

        for rank_idx, (doc, sc) in enumerate(ranked, 1):
            rrf_rows.append({
                "engine": "hybrid",
                "variant": "rrf",
                "query_id": qid,
                "rank": rank_idx,
                "doc_id": doc,
                "score": f"{sc:.6f}",
            })

        pool = pool_docs[qid]
        pool_values = [labels.get((qid, id)) for id in pool]
        relevant = sum(v == 1 for v in pool_values)
        values = [labels.get((qid, doc)) for doc, _ in ranked]

        # Precision
        p1 = sum(v == 1 for v in values[:1]) / 1.0
        p3 = sum(v == 1 for v in values[:3]) / 3.0
        p5 = sum(v == 1 for v in values[:5]) / 5.0
        p10 = sum(v == 1 for v in values[:10]) / 10.0
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        p10_list.append(p10)

        # NDCG
        for cutoff, clist in [(1, ndcg1_list), (3, ndcg3_list), (5, ndcg5_list), (10, ndcg10_list)]:
            dcg = sum((v == 1) / math.log2(rank + 1) for rank, v in enumerate(values[:cutoff], 1))
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant, cutoff) + 1))
            clist.append(dcg / ideal if ideal else 0.0)

        # MRR
        rr = next((1 / i for i, v in enumerate(values, 1) if v == 1), 0.0)
        mrr_list.append(rr)

        per_query_metrics.append({
            "query_id": qid,
            "p@5": p5,
            "p@10": p10,
            "ndcg@10": ndcg10_list[-1],
            "rr@10": rr,
        })

    # Write rrf.csv
    with rrf_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "variant", "query_id", "rank", "doc_id", "score"])
        writer.writeheader()
        writer.writerows(rrf_rows)

    summary = {
        "engine": "hybrid",
        "variant": "rrf",
        "k": k,
        "w_dense": w_dense,
        "w_bm25f": w_bm25f,
        "queries": len(queries),
        "precision@1": float(np.mean(p1_list)),
        "precision@3": float(np.mean(p3_list)),
        "precision@5": float(np.mean(p5_list)),
        "precision@10": float(np.mean(p10_list)),
        "ndcg@1": float(np.mean(ndcg1_list)),
        "ndcg@3": float(np.mean(ndcg3_list)),
        "ndcg@5": float(np.mean(ndcg5_list)),
        "ndcg@10": float(np.mean(ndcg10_list)),
        "mrr@10": float(np.mean(mrr_list)),
    }

    print("=" * 60)
    print("HYBRID RECIPROCAL RANK FUSION (RRF) EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Formulation: {w_dense}*Dense + {w_bm25f}*BM25F (k={k})")
    print(f"Queries: {summary['queries']}")
    print(f"Precision@5:  {summary['precision@5']:.4f}")
    print(f"Precision@10: {summary['precision@10']:.4f}")
    print(f"NDCG@10:      {summary['ndcg@10']:.4f}")
    print(f"MRR@10:       {summary['mrr@10']:.4f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    bm25f_p = base_dir / "evaluation" / "results" / "classical" / "bm25f.csv"
    dense_p = base_dir / "evaluation" / "results" / "neural" / "dense.csv"
    judg_p = base_dir / "evaluation" / "data" / "judgments_ensemble.csv"
    out_d = base_dir / "evaluation" / "results" / "hybrid"

    run_rrf(bm25f_p, dense_p, judg_p, out_d, k=20, w_dense=0.7, w_bm25f=0.3)
