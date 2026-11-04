# Data Collection & Corpus Placement

This component contains the Common Crawl collection, validation, and repair harness.

> [!NOTE]
> Raw data collection and creation of `technical_web_corpus.zip` is a separate external preparation step. It is **not** required for running the core IR benchmark pipeline.

---

## 1. Quick Setup: Placing the Corpus Archive

The pipeline assumes the raw corpus archive has already been created. Simply place `technical_web_corpus.zip` into either location:

* `data/00_corpus/technical_web_corpus.zip` (recommended)
* Or directly at the repository root: `technical_web_corpus.zip`

Once placed, you can run the full pipeline without manually executing collection:
```bash
# Build filters, classical index, and neural engine (excluding evaluation)
python run_full_pipeline.py --stage up-to-engine

# Or run the entire end-to-end benchmark
python run_full_pipeline.py
```

---

## 2. Advanced: Running Collection from Common Crawl

If you wish to crawl fresh WARC segments from Common Crawl from scratch:
```bash
cd data
python run_collect.py --step collect
python run_collect.py --step validate
python run_collect.py --step repair
```
`run_collect.py` will download segments into `data/00_corpus/` and record checkpoints in `data/runtime/` (both ignored by Git).
