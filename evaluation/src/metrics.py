"""Calculate binary relevance metrics from the shared judged pool."""

import math
from collections import defaultdict

CUTOFFS = (1, 3, 5, 10)
METRICS = [
    f"{name}@{k}"
    for k in CUTOFFS
    for name in ("precision", "pooled_recall", "pooled_f1", "pooled_ndcg")
] + ["pooled_ap@10", "rr@10"]


def calculate(runs, labels, query_types=None):
    query_types = query_types or {}
    pools = defaultdict(set)
    for rows in runs.values():
        for row in rows:
            pools[row["query_id"]].add(row["doc_id"])
    for (query_id, doc_id), value in labels.items():
        if value is not None:
            pools[query_id].add(doc_id)
    details = []
    for (engine, variant), rows in runs.items():
        queries = defaultdict(list)
        for row in rows:
            queries[row["query_id"]].append(row)
        for qid, results in queries.items():
            results = sorted(results, key=lambda r: int(r["rank"]))[:10]
            pool = pools[qid]
            pool_values = [labels.get((qid, id)) for id in pool]
            complete = all(v is not None for v in pool_values)
            relevant = sum(v >= 1 for v in pool_values if v is not None)
            values = [labels.get((qid, r["doc_id"])) for r in results]
            row = dict(
                engine=engine,
                variant=variant,
                query_id=qid,
                query_type=query_types.get(qid, "unspecified"),
                pool_size=len(pool),
                pool_judged=sum(v is not None for v in pool_values),
                pool_relevant=relevant if complete else None,
                pool_complete=int(complete),
                top10_judged=sum(v is not None for v in values),
            )
            for k in CUTOFFS:
                judged = all(v is not None for v in values[:k])
                hits = sum(v >= 1 for v in values[:k] if v is not None)
                precision = hits / k if judged else None
                recall = (
                    hits / relevant
                    if complete and relevant
                    else (0.0 if complete else None)
                )
                row[f"precision@{k}"] = precision
                row[f"pooled_recall@{k}"] = recall
                row[f"pooled_f1@{k}"] = (
                    (
                        2 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0.0
                    )
                    if complete
                    else None
                )
                dcg = sum(
                    (1 if v and v >= 1 else 0) / math.log2(rank + 1)
                    for rank, v in enumerate(values[:k], 1)
                )
                ideal = sum(
                    1 / math.log2(rank + 1) for rank in range(1, min(relevant, k) + 1)
                )
                row[f"pooled_ndcg@{k}"] = (
                    (dcg / ideal if ideal else 0.0) if complete else None
                )
            if all(v is not None for v in values):
                row["rr@10"] = next(
                    (1 / i for i, v in enumerate(values, 1) if v and v >= 1), 0.0
                )
            else:
                row["rr@10"] = None
            hits = 0
            score = 0.0
            for i, v in enumerate(values, 1):
                if v and v >= 1:
                    hits += 1
                    score += hits / i
            row["pooled_ap@10"] = (
                (score / relevant if relevant else 0.0) if complete else None
            )
            details.append(row)
    summary = []
    for engine, variant in runs:
        rows = [r for r in details if r["engine"] == engine and r["variant"] == variant]
        result = dict(
            engine=engine,
            variant=variant,
            query_type="all",
            queries=len(rows),
            complete_pools=sum(r["pool_complete"] for r in rows),
        )
        for name in METRICS:
            values = [r[name] for r in rows]
            output = (
                "pooled_MAP@10"
                if name == "pooled_ap@10"
                else ("MRR@10" if name == "rr@10" else name)
            )
            result[output] = (
                sum(values) / len(values)
                if values and all(v is not None for v in values)
                else None
            )
        summary.append(result)
    for engine, variant in runs:
        for query_type in sorted(set(query_types.values())):
            rows = [
                r for r in details
                if r["engine"] == engine
                and r["variant"] == variant
                and r["query_type"] == query_type
            ]
            if not rows:
                continue
            result = dict(
                engine=engine,
                variant=variant,
                query_type=query_type,
                queries=len(rows),
                complete_pools=sum(r["pool_complete"] for r in rows),
            )
            for name in METRICS:
                values = [r[name] for r in rows]
                output = (
                    "pooled_MAP@10"
                    if name == "pooled_ap@10"
                    else ("MRR@10" if name == "rr@10" else name)
                )
                result[output] = (
                    sum(values) / len(values)
                    if values and all(v is not None for v in values)
                    else None
                )
            summary.append(result)
    return details, summary


def calculate_graded(runs, labels, grades):
    """Compute binary and graded metrics from the shared judged pool."""
    pools = defaultdict(set)
    for rows in runs.values():
        for row in rows:
            pools[row["query_id"]].add(row["doc_id"])
    for (query_id, doc_id), value in labels.items():
        if value is not None:
            pools[query_id].add(doc_id)

    details = []
    summary = []
    for (engine, variant), rows in runs.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["query_id"]].append(row)
        run_metrics = []
        for query_id, pool in sorted(pools.items()):
            results = sorted(
                grouped[query_id], key=lambda row: int(row["rank"])
            )[:10]
            pool_binary = [labels.get((query_id, doc_id)) for doc_id in pool]
            if any(value is None for value in pool_binary):
                raise ValueError(f"Unjudged query pool: {query_id}")
            pool_grades = [grades.get((query_id, doc_id)) for doc_id in pool]
            if any(value is None for value in pool_grades):
                raise ValueError(f"Missing relevance grade in query pool: {query_id}")

            binary = [labels[(query_id, row["doc_id"])] for row in results]
            ranked_grades = [grades[(query_id, row["doc_id"])] for row in results]
            relevant_total = sum(value >= 1 for value in pool_binary)
            hits5 = sum(value >= 1 for value in binary[:5])
            hits10 = sum(value >= 1 for value in binary[:10])
            p5 = hits5 / 5
            p10 = hits10 / 10
            recall10 = hits10 / relevant_total if relevant_total else 0.0
            rr10 = next(
                (1 / rank for rank, value in enumerate(binary, 1) if value >= 1),
                0.0,
            )
            binary_dcg = sum(
                (value >= 1) / math.log2(rank + 1)
                for rank, value in enumerate(binary[:10], 1)
            )
            binary_ideal = sum(
                1 / math.log2(rank + 1)
                for rank in range(1, min(relevant_total, 10) + 1)
            )
            binary_ndcg10 = binary_dcg / binary_ideal if binary_ideal else 0.0

            pool_gain = sorted((2**int(value) - 1 for value in pool_grades), reverse=True)
            graded_ndcg = {}
            for cutoff in (5, 10):
                dcg = sum(
                    (2**int(value) - 1) / math.log2(rank + 1)
                    for rank, value in enumerate(ranked_grades[:cutoff], 1)
                )
                ideal = sum(
                    gain / math.log2(rank + 1)
                    for rank, gain in enumerate(pool_gain[:cutoff], 1)
                )
                graded_ndcg[cutoff] = dcg / ideal if ideal else 0.0

            detail = {
                "engine": engine,
                "variant": variant,
                "query_id": query_id,
                "p5": f"{p5:.4f}",
                "p10": f"{p10:.4f}",
                "recall10": f"{recall10:.4f}",
                "rr10": f"{rr10:.4f}",
                "binary_ndcg10": f"{binary_ndcg10:.4f}",
                "graded_ndcg5": f"{graded_ndcg[5]:.4f}",
                "graded_ndcg10": f"{graded_ndcg[10]:.4f}",
            }
            details.append(detail)
            run_metrics.append(
                (p5, p10, recall10, rr10, binary_ndcg10, graded_ndcg[5], graded_ndcg[10])
            )

        means = [sum(row[index] for row in run_metrics) / len(run_metrics) for index in range(7)]
        summary.append(
            {
                "engine": engine,
                "variant": variant,
                "P@5": means[0],
                "P@10": means[1],
                "Recall@10": means[2],
                "MRR@10": means[3],
                "Binary_NDCG@10": means[4],
                "Graded_NDCG@5": means[5],
                "Graded_NDCG@10": means[6],
            }
        )
    return details, summary
