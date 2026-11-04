# Evaluation

Evaluation reads the fixed query set and exported classical/neural results,
serves the relevance-labeling UI, and writes pooled metrics. Runtime queries,
results, and logs are ignored by Git; `data/judgments.csv` is the tracked
human-label input.

The active benchmark uses ten natural-language queries in
`queries/final_queries.jsonl`. The store starts successfully when result
snapshots are missing and reports those engine families as pending.

```bash
cd evaluation
python run_evaluation.py --prepare
python run_evaluation.py --metrics
# Plumbing-only test metrics with seeded synthetic 0/1 labels:
python run_evaluation.py --metrics --synthetic-random --seed 20261102
python run_evaluation.py --host 127.0.0.1 --port 8001
```

The corpus archive defaults to
`../filter/outputs/stage_05_archive/technical_web_corpus.zip`. Metrics are written atomically to `results/metrics.csv` and
`results/per_query.csv`; both include `query_type`, and the summary contains
one `all` row plus separate rows for each query type. Synthetic labels are
written to timestamped `results/synthetic_random_judgments_*.csv` files and
are explicitly marked as `synthetic_random_relevance`.
Evaluation logs append to `logs/evaluation.log`.

For any corpus run, use the same evaluator with that run's archive and
generated result snapshots:

```bash
python run_evaluation.py \
  --metrics --force --synthetic-random --seed 20261102 \
  --corpus-zip ../filter/outputs/stage_05_archive/technical_web_corpus.zip
```

This writes `results/metrics.csv` with an `all` aggregate and one row per
query type, plus `results/per_query.csv` for individual queries. Synthetic
labels are only a plumbing check; replace them with reviewed judgments before
reporting benchmark quality.
