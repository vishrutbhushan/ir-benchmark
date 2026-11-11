# Data Fetch & Corpus Creation

A single-step, self-contained pipeline for fetching Common Crawl WARC segments and packaging the production corpus.

The output is written directly to **`data/00_corpus/technical_web_corpus.zip`**, making it immediately usable by the main benchmark pipeline (`run_full_pipeline.py`) without any copying, moving, or manual repair steps.

---

## 1. Quick Start: Use Existing Corpus

If you already have `technical_web_corpus.zip`, place it in `data/00_corpus/` (or directly in the repository root). You can immediately run the main pipeline:

```bash
python run_full_pipeline.py
```

---

## 2. Generating the Corpus from Scratch

To fetch webpages from Common Crawl and build the corpus archive:

```bash
# Run data collection (fetches HTML, writes manifest, and builds the ZIP directly)
python data/run_collect.py

# Optional: customize concurrency or download a subset
python data/run_collect.py --workers 8 --max-docs 1000
```

### What `data/run_collect.py` Does:
1. Reads candidate entries from `data/00_corpus/metadata/corpus_manifest.jsonl`.
2. Downloads original HTML bodies via parallel HTTP Range requests from Common Crawl.
3. Saves raw HTML files directly to `data/00_corpus/documents/html/<technology>/`.
4. Writes `data/00_corpus/metadata/corpus_manifest.jsonl` and `corpus_state.json`.
5. Automatically packages everything into `data/00_corpus/technical_web_corpus.zip`.

Once complete, `python run_full_pipeline.py` or `python filter/run_filter.py` can be executed immediately with zero manual intervention.
