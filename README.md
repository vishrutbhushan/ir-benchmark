# Technical corpus — data branch

Owns collection, validation, live repairs, English/content filtering and exact URL-alias exclusions. Both IR engines consume the same corrected dataset and document IDs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --step all --archive original_corpus.zip
```

For new downloads, supply the existing Common Crawl candidate JSONL:
`python run_pipeline.py --step collect --candidates common_crawl_candidates.jsonl`.

After collection, `--step all` automatically snapshots the downloaded folder when no original archive is supplied. Recovery copies stay local.

Steps are in `src/step_01_collection` through `src/step_05_archive`.
Final output: `outputs/step_05_archive/technical_web_corpus.zip`, containing corrected HTML and metadata. ZIPs, raw HTML and recovery copies stay out of Git. To package an already corrected folder, run `python run_pipeline.py --step pack --corpus technical_web_corpus`.
