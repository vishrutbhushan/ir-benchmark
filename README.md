# Technical IR Benchmark

Course project comparing BM25/TF-IDF with neural retrieval on technical HTML
pages. This branch implements neural retrieval; teammates maintain classic ranking.

## Setup and run

Use Python 3.12+ and one environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-technical-web-corpus.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Install the PyTorch build
appropriate for your platform/GPU if the default installation does not provide CUDA.

Obtain the team's corrected `technical_web_corpus/` folder, including
`metadata/corpus_manifest.jsonl`. Alternatively, put the original
`technical_web_corpus.zip` in the repository root; initial preparation validates,
repairs and extracts it. Live repairs may differ from the team's corrected snapshot,
so both engines must use the same corrected folder for a fair comparison.

```bash
python run_pipeline.py --phase all --workers 8 --batch-size 512
python run_pipeline.py --phase serve
```

Open **http://127.0.0.1:8000** for search, retrieval-method selection, filters and
pagination. Build before serving. Use `--device cpu` on a system without CUDA;
it will be slower. No database server is needed.

`all` runs corpus validation/fixes → extraction → four-zone windows → embeddings
→ learned weights → classifiers → routing → verification. It also evaluates when
`data/evaluation/qrels.csv` exists. A full build is required before search/serve.

Interrupt with Ctrl+C to pause. Run the same command to resume: unchanged stages
and completed zones are reused, and partial embeddings resume at their checkpoint.
Keep the corpus, downloaded model and `outputs/` intact. Changes to embedding inputs
or the encoder can invalidate affected vectors. Use fewer `--workers` to reduce
CPU preprocessing load without changing index contents.

## What is included

- Independent title, headings, body and code indexes; visible table text joins body.
- Pinned Qwen3-Embedding-0.6B and hand-coded cosine ranking.
- Plain dense baseline, learned weighted zones, centroid and Naive Bayes class routing.
- Durable SQLite text/metadata, NumPy vectors and bounded search caches.
- Learning CSVs: 44 development queries / 262 relevance pairs for zone weights;
  420 documents / 14 source-domain classes for classification (280/70/70 split).
- Twenty challenge queries and hand-coded NDCG@5/MAP evaluation.

The full four-zone build and evaluation have **not yet completed**. Generated
outputs, checkpoints, weights, models, raw HTML and logs are excluded from Git.
A fresh clone therefore needs the corpus and a local build. Training refreshes
CSV features/text against that build; committed feature values are not final results.

## Checks and evaluation

```bash
python -m unittest discover -s tests -v
python run_pipeline.py --phase audit
python run_pipeline.py --phase verify
python run_pipeline.py --phase search --query "PostgreSQL JSONB indexing" --mode dense --limit 10
python run_pipeline.py --phase evaluation-pool
python run_pipeline.py --phase evaluate
```

`audit` needs extracted data and refreshed learning tables; `verify` and the search/
evaluation commands need a completed build. `evaluation-pool` exports candidate
source excerpts for relevance judgments. `evaluate` requires graded
`data/evaluation/qrels.csv`; final judgments are not included yet. To compare engines,
add `--classic-rankings PATH` using the shared query suite and judgments.

See [docs/NEURAL.md](docs/NEURAL.md) for architecture, data and course connections.
Generated reports/charts are under `outputs/neural/`; stage logs are under `logs/neural/`.
