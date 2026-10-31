# IR Benchmark

Unified workspace for collecting and filtering technical documents, running
classical BM25 retrieval, running neural retrieval, and evaluating both
systems. The `main` branch contains code and small seed files; large corpus
data, model artifacts, indexes, logs, and generated results stay local.

## Layout

| Directory | Responsibility | Runtime data |
|---|---|---|
| `data/` | Common Crawl collection, validation, and repair | `data/00_corpus/`, `data/runtime/` |
| `filter/` | Shared extraction, quality filtering, version deduplication, ZIP packaging | `filter/outputs/`, `filter/logs/` |
| `classical/` | Tokenization, BM25/BM25F, multi-level retrieval | `classical/outputs/`, `classical/logs/` |
| `neural/` | Windowing, embeddings, learning, routing, search, UI | `neural/outputs/`, `neural/results/`, `neural/logs/` |
| `evaluation/` | Judgments, comparison UI, and metrics | `evaluation/queries/`, `evaluation/results/`, `evaluation/logs/` |
| `data_inputs/` | Small versioned neural training seeds | tracked |

## Setup

Use Python 3.11+ and install the combined dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Each component also has a focused requirements file. Run commands from the
component directory so its `src` package and relative runtime paths resolve.

## Staged pipeline

Stages write durable outputs and completion markers. Re-running a command
reuses completed stages; add `--force` when inputs or code have changed.

```bash
cd data
python run_collect.py --step all

cd ../filter
python run_filter.py --step all

cd ../classical
python run_pipeline.py --phase all

cd ../neural
python run_pipeline.py --step all

cd ../evaluation
python run_evaluation.py --metrics

cd ..
python run_search_ui.py
```

The shared archive is
`filter/outputs/stage_05_archive/technical_web_corpus.zip`. Classical reads
structured records from `filter/outputs/stage_04_filtering/`; neural reads the
shared archive and stores stage outputs under `neural/outputs/`.

To continue after interruption, rerun the same command. To execute a later
stage only, use `--phase 2`/`--phase 3` for classical or `--step 01`–`--step 09`
for neural. Logs are appended under each component's `logs/` directory.

The benchmark suite contains ten natural-language developer queries in
`evaluation/queries/final_queries.jsonl` and
`data_inputs/benchmark_queries.jsonl`. Evaluation starts with pending result
families when exports have not yet been generated.

## Git and data policy

`data/00_corpus/`, runtime queries, archives, outputs, indexes, databases,
vectors, model files, logs, and evaluation exports are ignored by Git. Do not
force-add them. Only source code, documentation, configuration, judgments, and
small training inputs belong in commits.
