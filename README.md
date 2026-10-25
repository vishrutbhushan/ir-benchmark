# Neural technical-document search

This branch owns neural retrieval. `data` owns collection/filtering and the corpus ZIP; `evaluation` owns final benchmark evaluation. BM25 stays on its own branch.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --step all
python run_pipeline.py --step 09
```

Put the prepared ZIP at `data/00_corpus/technical_web_corpus.zip`, or its extracted folder alongside it. Open **http://127.0.0.1:8000**. Configure `config.json`; explicit CLI options override run defaults. Use `--device cpu` without CUDA. Compatible completed embeddings are reused.

| Step | Purpose |
|---|---|
| 01 | Verify prepared corpus and extract text |
| 02 | Build title/headings/body/code windows |
| 03 | Generate persistent Qwen3 embeddings |
| 04 | Learn global zone weights using squared error |
| 05 | Train content centroids and categorize every document |
| 06 | Build category-to-window row lists |
| 07 | Verify full-corpus coverage, cosine, ranking and pagination |
| 08 | Run a search from the command line |
| 09 | Serve search UI and saved HTML documents |

```bash
python run_pipeline.py --step 08 --mode dense --query "PostgreSQL JSONB indexing"
python run_pipeline.py --step audit
python run_pipeline.py --step references
python -m unittest discover -s tests -v
```

Steps 01–03 check preceding stages; later steps require earlier artifacts. Default search routes to the nearest **3 content classes**, configurable. `dense` and `zones` search exhaustively. Query text resembling Boolean syntax is embedded; logical operators are not enforced.

## Files

- `data/step_04_weights/`: 240 judged pairs, 60 query formulations / 20 intents; training and validation stay separate by intent.
- `data/step_05_classification/`: 168 content seeds, 14 classes, 10 training + 2 validation documents per class; review decisions are retained.
- Reviewed labels are inputs and are never overwritten by pipeline runs. Local `documents/` reference folders are ignored.
- `results/`: small reports and actual cosine feature tables.
- `outputs/`: ignored databases, model weights, vectors and category row lists.
- `logs/`: ignored run logs.

The corpus has **68,414 documents and 476,906 window vectors**. Click a result to open its saved HTML; original URLs are secondary links. Scripts are disabled in the saved-document viewer; external assets may be unavailable.

Model/window changes require compatible embeddings. Weight and category changes reuse existing window vectors. Do not add corpus files or embeddings to Git.

See [docs/NEURAL.md](docs/NEURAL.md) for architecture and learning details. If VS Code shows stale Git changes, use **Git: Refresh** or **Developer: Reload Window**.
