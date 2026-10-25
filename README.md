# Neural technical-document retrieval

This branch contains the neural engine only. `data` owns collection/filtering and the corrected corpus ZIP. `evaluation` owns benchmark queries. BM25 work stays on its own branch.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the corrected ZIP from `data` at `data/00_corpus/technical_web_corpus.zip`, or place its extracted `technical_web_corpus/` folder there. Then:

```bash
python run_pipeline.py --step all
python run_pipeline.py --step 09
```

Open **http://127.0.0.1:8000**. Use `--device cpu` without CUDA. To limit preprocessing load use `--workers 4`. Ctrl+C pauses embedding; rerun the same command to reuse completed stages and checkpoints.

| Step | Source folder | Purpose |
|---|---|---|
| 01 | `src/step_01_extraction/` | Verify prepared inputs and extract structured text |
| 02 | `src/step_02_windows/` | Build title/headings/body/code token windows |
| 03 | `src/step_03_embeddings/` | Generate durable Qwen3 embeddings |
| 04 | `src/step_04_weights/` | Learn zone weights from relevance labels |
| 05 | `src/step_05_classification/` | Train Naive Bayes and neural centroids |
| 06 | `src/step_06_routing/` | Tune category-based candidate selection |
| 07 | `src/step_07_verification/` | Check coverage, vectors, rankings and learning data |
| 08 | `src/step_08_search/` | Hand-coded cosine retrieval |
| 09 | `src/step_09_ui/` | Search, filters, snippets and pagination |

Run any numbered step with `--step NN`. Steps 01–03 reuse their preceding build artifacts; steps 04 onward require the earlier stages.

```bash
python run_pipeline.py --step 08 --mode dense --query "PostgreSQL JSONB indexing"
python run_pipeline.py --step audit
python run_pipeline.py --step references
python -m unittest discover -s tests -v
```

## Files

- `data/step_04_weights/`: 262 relevance pairs / 44 development queries and locally generated source documents.
- `data/step_05_classification/`: 420 compact document/class labels, 14 categories, technology mapping and local source references. Full training text comes from SQLite.
- Source-document `documents/` folders are local and Git-ignored; regenerate them with `--step references` after extraction.
- `results/step_*/`: small versioned metrics, charts and HTML reports.
- `outputs/step_*/`: ignored working databases, model files and vectors.
- `logs/`: ignored run logs.

The current build covers **68,414 documents / 476,906 vectors**. Dense, learned-zone and NB/centroid-routed modes are implemented. Learned weights currently select body-only scoring, and routing retains most classes; improvements are not assumed. Final comparative evaluation belongs to the evaluation branch.

See [docs/NEURAL.md](docs/NEURAL.md) for design details.
