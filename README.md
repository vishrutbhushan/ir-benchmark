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

Put the prepared ZIP at `data/00_corpus/technical_web_corpus.zip`, or its extracted folder alongside it. Search reads HTML directly from the ZIP. Builds remove temporary HTML after extraction and reuse an unchanged ZIP. Open **http://127.0.0.1:8000**. Configure `config.json`; explicit CLI options override run defaults. Use `--device cpu` without CUDA. The pipeline reuses compatible embeddings.

| Step | Purpose |
|---|---|
| 01 | Verify prepared corpus and extract text |
| 02 | Build title/headings/body/code windows |
| 03 | Generate persistent Qwen3 embeddings |
| 04 | Load manual zone weights from config.json |
| 05 | Train content centroids and categorize every document |
| 06 | Build category-to-window row lists |
| 07 | Verify full-corpus coverage, cosine, ranking and pagination |
| 08 | Run a search from the command line |
| 09 | Serve document links and saved HTML |

```bash
python run_pipeline.py --step 08 --mode dense --query "your query"
python run_pipeline.py --step audit
```

Steps 01-03 check preceding stages; later steps require earlier artifacts. Default search routes to the nearest **3 content classes**, configurable. `dense` and `zones` search exhaustively. Query text resembling Boolean syntax is embedded; logical operators are not enforced.

Logs append to `logs/step_NN/run.log` or `logs/all/run.log`. They record stage times, progress, errors and tracebacks. Ctrl+C preserves completed embedding checkpoints.

## Files

- `data/step_04_weights/`: 63 judged pairs across 16 distinct queries, including code snippets and a command; 12 queries train weights and 4 validate them.
- `data/step_05_classification/`: 168 content seeds, 14 classes, 10 training + 2 validation documents per class; the review CSV records label decisions.
- The pipeline reads reviewed labels without overwriting them. Read saved documents through the search UI.
- `results/`: learning summaries, charts and actual cosine features.
- `outputs/`: ignored databases, model weights, vectors and category row lists.
- `logs/`: ignored run logs.

The corpus has **68,414 documents and 476,906 window vectors**. Each result shows its title, technology, assigned categories and scores. Original and saved-HTML links open in new tabs. The saved-document viewer disables scripts. Some external assets may be unavailable.

Model/window changes require compatible embeddings. Weight and category changes reuse existing window vectors. Do not add corpus files or embeddings to Git.

See [docs/NEURAL.md](docs/NEURAL.md) for architecture and learning details. If VS Code shows stale Git changes, use **Git: Refresh** or **Developer: Reload Window**.

Set `result_limit` (default 10) and `page_size` (default 5) in `config.json`. The UI starts with an empty search box and has no technology/category filters or document snippets.

Ranking reads manual `zone_weights` from `config.json`: title 0.15, headings 0.15, body 0.50, code 0.20. They must be positive and sum to one. Restart search after changing them. Existing learning results remain available as previous experiments.
