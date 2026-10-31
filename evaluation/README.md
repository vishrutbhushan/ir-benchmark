# Evaluation

Evaluation reads the fixed query set and exported classical/neural results,
serves the relevance-labeling UI, and writes pooled metrics. Runtime queries,
results, and logs are ignored by Git; `data/judgments.csv` is the tracked
human-label input.

```bash
cd evaluation
python run_evaluation.py --prepare
python run_evaluation.py --metrics
python run_evaluation.py --host 127.0.0.1 --port 8001
```

The corpus archive defaults to
`../filter/outputs/stage_05_archive/technical_web_corpus.zip`. Metrics are
written atomically to `results/metrics.csv` and `results/per_query.csv`.
Evaluation logs append to `logs/evaluation.log`.
