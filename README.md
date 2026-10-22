# Technical IR Benchmark

Course project comparing classic BM25/TF-IDF with neural retrieval on technical
HTML documents. This work implements the neural side; teammates own classic ranking.
Development branch: `develop_vishrut`.

## Run

Use the single `.venv` and existing requirements file:

```bash
source .venv/bin/activate
python -m pip install -r requirements-technical-web-corpus.txt
python run_pipeline.py --phase serve
```

Open **http://127.0.0.1:8000** for search, filters and pagination. The current local
index contains 68,418 documents / 767,679 passages, with 1,024-dimensional vectors
stored on disk (~1.46 GiB). Search uses hand-coded cosine and max-passage document
ranking. The summary hierarchy is removed. No database server is required; SQLite
stores source text/metadata and NumPy files store vectors. Bounded caches speed reuse.

Build, check or search from the CLI:

```bash
python run_pipeline.py --phase all --workers 8 --batch-size 512
python run_pipeline.py --phase validate --workers 8
python run_pipeline.py --phase search --query "PostgreSQL JSONB indexing" --limit 10
python run_pipeline.py --phase verify
python -m unittest discover -s tests -v
```

`all`: validate/recover sources → fix the working dataset → extract → chunk → embed
→ verify. Completed unchanged stages are reused; interrupted embedding builds resume.
Individual stages: `prepare` (including shared corpus fixes), `extract`, `chunk`, `embed`.
The model is pinned Qwen3-Embedding-0.6B.

## Shared corpus

Both teammates use **`technical_web_corpus/` directly**, including its normal
`metadata/corpus_manifest.jsonl`. Accepted repairs are applied to the HTML files;
excluded captures are outside the working dataset. No separate allowlist is needed.
The corrected dataset contains 68,418 documents, across 11 categories / 76 technologies.
The original 79,268-document ZIP remains unchanged; local recovery copies and original
metadata are in `outputs/corpus_backup/`. Full hash/coverage checks verify the corrected
files; this is not a claim of perfect content extraction.

Neural preprocessing is separate from classic preprocessing. It preserves negation,
case, versions, identifiers and code whitespace, using the model's tokenizer.

## Learning data

| Task | Self-contained CSV | Size |
|---|---|---|
| Zone weights | `data/neural/zone_weights/training.csv` | 44 queries, 264 pairs; 262 usable |
| NB and centroid classification | `data/neural/classification/training.csv` | 420 documents; 14 classes |

The first table includes query/document details, relevance and four zone cosines.
The second includes full document text, class ID/name and split. Use only its `text`
column as classifier input. Both classifiers share its 280/70/70 train/validation/test
split. No CSV joins or user annotation are needed. Class labels describe source domains.

```bash
python run_pipeline.py --phase development-features --batch-size 32
python run_pipeline.py --phase classification-data
```

These commands prepare learning data. Weight fitting, classifier training/routing
and separate production zone indexes remain pending. The four development zones
are title, headings, body and code; table text joins body. The book starts with
Boolean zone matches; our cosine features are a neural adaptation. See
[design and book alignment](docs/DESIGN.md).

## Files and results

- `src/corpus_validation/`: archive checks and fixes to the shared working dataset.
- `src/neural/preprocessing/`: independent structured extraction and token windows.
- `src/neural/learning/`: separate zone-feature and classification-data modules.
- `src/neural/search.py`, `server.py`, `web/`: cosine ranking and local UI.
- `outputs/`: corrected-corpus checks, extraction, durable model/index, charts and dashboards.
- `logs/`: stage logs. Both output folders are Git-ignored.

[Verification report](outputs/neural/verification/report.md) ·
[Corpus-fix report](outputs/corpus_fix/report.md) ·
[Development report](outputs/neural/development/report.md) ·
[Classification report](outputs/neural/classification/report.md) ·
[History](docs/PROJECT_HISTORY.md)

The 25 candidate queries in `queries.jsonl` support engineering verification.
Final 20-query selection, relevance judgments, hand-coded NDCG@5/MAP and comparative
failure analysis remain evaluation work. No relevance improvements are claimed yet.

Original crawler (only for intentionally collecting a new corpus):
`python build_technical_web_corpus.py --crawl CC-MAIN-2026-34`.
Do not resume it into the corrected working corpus.
