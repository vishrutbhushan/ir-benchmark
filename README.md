# IR Benchmark

This repository combines corpus collection and filtering with classical BM25,
neural retrieval, and relevance evaluation.

## Pipeline order

1. Run `data/run_collect.py` to collect, validate, and repair the source corpus.
2. Run `filter/run_filter.py --step all` to produce the shared corpus archive.
3. Run `classical/run_pipeline.py --phase all` for BM25 retrieval.
4. Run `neural/run_pipeline.py --step all` for neural retrieval.
5. Run `evaluation/run_evaluation.py --metrics` to compute comparison metrics.

The `filter/` output is the shared input for both retrieval pipelines. Runtime
outputs, model artifacts, archives, and databases are intentionally ignored.
