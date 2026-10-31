# Data collection

This component owns Common Crawl collection, validation, and repair. It writes
the working corpus to `data/00_corpus/` and checkpoints to `data/runtime/`.
Those paths are intentionally ignored by Git.

Place an incoming archive at `data/00_corpus/technical_web_corpus.zip` before
running validation, or run the collection stage first.

```bash
cd data
python run_collect.py --step collect
python run_collect.py --step validate
python run_collect.py --step repair
```

`--step all` runs validation and repair together. Completed stages are marked
with `stage.done.json` and reused on the next run. Use `--force` to rerun a
stage. Collection resumes from the downloader's manifest checkpoints.
