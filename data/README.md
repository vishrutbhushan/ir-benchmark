# Data references

The source corpus is a supplied Common Crawl HTML archive:

```text
data/00_corpus/technical_web_corpus.zip
```

Collection code is maintained on the repository's `data` branch. The source manifest records original URLs, crawl identifiers, WARC filenames, byte offsets, record lengths and HTML hashes. The main pipeline starts from this archive.

Filtering writes the prepared shared archive to:

```text
filter/outputs/stage_05_archive/technical_web_corpus.zip
```

Both retrieval engines consume the same retained documents. The classical engine uses `filter/outputs/stage_04_filtering/structured_records.jsonl`; neural extraction reads the prepared HTML and keeps separate title, headings, body and code stores.

Inputs maintained in Git:

- `evaluation/queries/final_queries.jsonl`: the fixed query IDs and texts.
- `evaluation/data/judgments.csv`: the complete strict judgment history. Evaluation selects only pairs in the current six-engine pool.
- `evaluation/data/judgment_queries.json`: query texts bound to the judgment IDs.
- `evaluation/data/judge_decisions.csv`: new per-judge decisions, reasons and prompt hashes, saved after each response for resuming.
- `neural/data/zone_weight_queries.csv` and `zone_weight_judgments.csv`: the existing binary calibration dataset. Active ranking uses configured zone weights while `use_learned_zone_weights` is false.

The strict rubric is defined in `evaluation/src/judges.py`. It uses the same title, leading headings, body excerpt and code excerpt prompt for newly missing pairs. Existing labels are preserved.
