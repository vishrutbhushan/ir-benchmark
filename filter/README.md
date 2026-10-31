# Shared corpus filter

The filter is the single corpus boundary for both retrieval systems. It
extracts structured documents, applies language and URL-quality rules,
deduplicates versioned pages, applies reviewed exclusions, and packages the
shared ZIP.

```bash
cd filter
python run_filter.py --step all
```

Inputs default to `../data/00_corpus/`. Outputs are named by stage:

- `outputs/stage_02_validation/`
- `outputs/stage_03_repair/`
- `outputs/stage_04_filtering/`
- `outputs/stage_05_archive/technical_web_corpus.zip`

Each completed stage has a marker and is reused. Use `--force` after changing
filter code or input data. Logs append to `logs/filter.log`.
