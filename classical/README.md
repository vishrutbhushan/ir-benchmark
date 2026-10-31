# Classical retrieval

This component tokenizes the structured records produced by `filter/`, builds
an inverted index, and runs BM25, BM25F, and multi-level retrieval.

```bash
cd classical
python run_pipeline.py --phase all
```

Default input:
`../filter/outputs/stage_04_filtering/structured_records.jsonl`.
Default query input:
`../evaluation/queries/final_queries.jsonl`.

Outputs:

- `outputs/stage_02_tokenization/`
- `outputs/stage_03_retrieval/`
- `logs/stage_02_tokenization.log`
- `logs/stage_03_retrieval.log`

Use `--phase 2` or `--phase 3` to continue from a specific stage. Completed
stages are marked with `stage.done` and reused; use `--force` to rebuild them.
