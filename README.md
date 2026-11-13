# Technical Document Retrieval Benchmark

The pipeline compares six search methods over the same technical web corpus:
BM25, BM25F, dense neural, zone-weighted neural, cluster-routed neural, and
BM25F plus routed neural with reciprocal-rank fusion. Champion-list results are
excluded from the benchmark and search UI.

## Setup

Use Python 3.11 or newer and the single dependency file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the source corpus at `data/00_corpus/technical_web_corpus.zip`. Corpus
archives, extracted files, indexes, and embeddings are local runtime data and
are not tracked in Git.

## Run

All paths, models, retrieval weights, query limits, result variants, and UI
settings are in `config/pipeline.json`.

```bash
python run_full_pipeline.py
```

The pipeline runs filtering, BM25/BM25F indexing, neural indexing and result
export, then evaluation. It writes the six ranked result sets and binary and
graded metrics under `evaluation/results/`.

After indexes are ready, start the six-engine search UI:

```bash
python run_search_ui.py
```

Open `http://127.0.0.1:8000`. The UI returns at most ten results per query,
shows five per page, and links each result to its source.

## Evaluation and presentation

`evaluation/data/judgments.csv` contains the strict labels for the six-engine
top-ten pool. Binary NDCG@5 and NDCG@10 are the primary metrics; graded NDCG@10
is secondary. The champion-list run is not part of the pool.

The `presentation/` folder contains the updated deck, `queries.csv`,
`zone_weight_calculation.csv`, and the strict judgments for its pooled results.
The pre-edit deck is saved under `presentation/backups/`.
