# Versioned training inputs

This directory contains versioned query, benchmark, and relevance
inputs used by the active pipelines.

`neural_weights/zone_weight_queries.csv` and
`neural_weights/zone_weight_judgments.csv` are focused binary (`0`/`1`)
examples used only to learn continuous zone weights. They are intentionally
separate from benchmark queries.

`benchmark_queries.jsonl` is the canonical ten-query benchmark suite mirrored
by `evaluation/queries/final_queries.jsonl`.

`benchmark_queries_bm25_latest.jsonl` is a copied snapshot from the latest
BM25 branch query update and is kept for comparison only. It is not used for
zone-weight learning.
