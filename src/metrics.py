"""Calculate binary relevance metrics from the shared judged pool."""

import math
from collections import defaultdict

CUTOFFS = (1, 3, 5, 10)
METRICS = [
    f"{name}@{k}"
    for k in CUTOFFS
    for name in ("precision", "pooled_recall", "pooled_f1", "pooled_ndcg")
] + ["pooled_ap@10", "rr@10"]


def calculate(runs, labels):
    pools = defaultdict(set)
    for rows in runs.values():
        for row in rows:
            pools[row["query_id"]].add(row["doc_id"])
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
            relevant = sum(v == 1 for v in pool_values)
            values = [labels.get((qid, r["doc_id"])) for r in results]
            row = dict(
                engine=engine,
                variant=variant,
                query_id=qid,
                pool_size=len(pool),
                pool_judged=sum(v is not None for v in pool_values),
                pool_relevant=relevant if complete else None,
                pool_complete=int(complete),
                top10_judged=sum(v is not None for v in values),
            )
            for k in CUTOFFS:
                judged = all(v is not None for v in values[:k])
                hits = sum(v == 1 for v in values[:k])
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
                    (v == 1) / math.log2(rank + 1)
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
                    (1 / i for i, v in enumerate(values, 1) if v == 1), 0.0
                )
            else:
                row["rr@10"] = None
            hits = 0
            score = 0.0
            for i, v in enumerate(values, 1):
                if v == 1:
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
