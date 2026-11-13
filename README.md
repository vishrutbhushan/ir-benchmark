# Technical document retrieval

Python 3.12 or newer is required. Use one environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the corpus ZIP at `data/00_corpus/technical_web_corpus.zip`.
See [data references](data/README.md) for the corpus and annotation inputs.

For CUDA retrieval, use a CUDA-compatible PyTorch installation. The first neural build downloads the configured embedding model. If new judgments are needed, start Ollama with the two models listed in `evaluation.judges.models` installed and set `evaluation.judges.url` to its address. Existing judgments are reused without contacting Ollama when the result pool has no missing labels.

## Pipeline

```bash
python run_full_pipeline.py
```

Stages run sequentially:

1. `filter`: validate and repair HTML, filter content, remove version/URL aliases, and package the shared corpus.
2. `classical`: tokenize title, headings, body and code; build the positional inverted index; run BM25 and BM25F.
3. `neural`: extract the same documents, create overlapping zone windows, reuse or create embeddings, cluster documents and build routing postings.
4. `retrieve`: load complete indexes and the embedding model; export six ranked runs and measure backend retrieval latency.
5. `judge`: reuse strict labels and judge only missing query-document pairs, one model and one input at a time.
6. `evaluation`: calculate pooled metrics and generate charts from the saved runs and judgments.

Run a stage separately:

```bash
python run_full_pipeline.py --stage classical
python run_full_pipeline.py --stage neural
python run_full_pipeline.py --stage retrieve
python run_full_pipeline.py --stage judge
python run_full_pipeline.py --stage evaluation
```

Use `--config path/to/config.json` with the pipeline command to select another configuration. The same file is passed to each stage, including `--stage search-ui`.

Input fingerprints control reuse of completed artifacts. Retrieval captures exclude startup, UI rendering and saved-document loading. Evaluation uses one CPU thread, one GPU workload and sequential queries. The judge stages unload each model before loading the next.

## Search

```bash
python run_search_ui.py
```

Open `http://127.0.0.1:8000`. Select BM25, BM25F, dense, zone-weighted dense, cluster-routed dense, or Hybrid RRF. Results link to the source URL and saved HTML.

## Configuration

Edit `config/pipeline.json`:

| Section | Parameters |
|---|---|
| `paths` | Corpus inputs, stage outputs, query suite, judgment master and logs |
| `execution` | CPU thread limit |
| `filter` | Preprocessing workers |
| `classical` | Workers, BM25 k1/b, zone weights, zone length normalization, taxonomy boost, routing mode and retrieval depth |
| `neural` | Model/revision/hash, token window, overlap, device, batch size, workers, zone weights, learned-weight flag, clustering and membership settings, cache sizes, result limit and page size |
| `evaluation.latency` | Repetitions, warmup queries and single CPU thread |
| `evaluation.rrf` | Rank constant, engine weights and input depths |
| `evaluation.judges` | Ollama URL, strict judge models, retries, timeout and response token limit |

The embedding model and index must agree. Changing window or model settings requires compatible embeddings. Changing clustering settings rebuilds clusters and routing without rebuilding passage embeddings. New query text must use a new query ID to prevent reuse of judgments for a different query.

## Folders

```text
config/                 Shared pipeline settings
retrieval.py            Shared backend engine loading and retrieval
run_full_pipeline.py    Pipeline orchestration
run_search_ui.py        HTTP search interface
data/00_corpus/         Source ZIP and prepared working HTML
filter/src/             Shared corpus validation, repair and extraction
classical/src/          Tokenization, positional index and lexical scoring
neural/src/             Extraction, windows, embeddings, clustering and routing
neural/data/            Existing zone-weight calibration inputs
evaluation/queries/     Fixed query suite
evaluation/data/        Strict judgment history and resumable judge decisions
evaluation/src/         Retrieval capture, judging, metrics and charts
*/outputs/              Local corpus, index and embedding artifacts
neural/results/         Neural stage reports
evaluation/results/     Ranked runs, timings, metrics and charts
*/logs/                 Execution logs
```

Corpus files, indexes, model weights, embeddings, logs and local trash are ignored by Git.
