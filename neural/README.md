# Neural retrieval

This component consumes the shared filtered ZIP and runs extraction,
overlapping neural windows, Qwen3 embeddings, zone-weight learning,
unsupervised clustering, routing, verification, search, and the web UI.

```bash
cd neural
python run_pipeline.py --step all
python run_pipeline.py --step 08 --query "how do Kubernetes probes work"
python run_pipeline.py --step 09
```

The default corpus is
`../filter/outputs/stage_05_archive/technical_web_corpus.zip`; extracted
runtime data is under `../data/00_corpus/`. Outputs use `outputs/step_NN/`,
results use `results/`, and logs use `logs/step_NN/` or `logs/all/`.

Stages 01–07 write `stage.done` markers and are reused on subsequent runs.
Use `--step 03` to resume embedding generation or `--force` to rebuild a
completed stage. Interrupted embedding work retains its checkpoint files.
Set `--device cpu` when CUDA is unavailable.

Step 04 does not use the configured zone weights as its result. It writes
`results/step_04_weights/features/training_features.csv` from the focused,
graded judgments in `data_inputs/neural_weights/zone_weight_judgments.csv`,
fits query-balanced constrained weights, and writes the presentation-friendly
`weights.csv` and `outputs/step_04_weights/model.json`. Relevance labels are binary `0` or `1`; the fitted weights are continuous
positive values and are not labels. Corpus size is controlled by the same runner
and configuration mechanism for every run; corpus size never selects a
different implementation.

For any corpus run, point `--corpus-dir` and `--corpus-zip` at the selected
filtered corpus. The same stages and verification checks are used; only paths
and resource limits change. Step 05 creates and persists one mean-normalized
Qwen3-Embedding-4B vector per available zone. Zones larger than the 4B
context window are token-chunked and their 4B chunk vectors are mean-pooled
back into that one zone vector; no zone is silently truncated.
in `outputs/step_05_clustering/zone_vectors.npz`, preserves zone identity and
presence, and clusters documents with deterministic NumPy k-means. The
window-level Qwen3-Embedding-0.6B vectors remain in
`step_03_embeddings/<zone>/vectors.npy` for passage-level retrieval; they are
not replaced by the zone vectors. Step 06
stores cluster postings. Routed search embeds the query with both models,
selects the configured top-N clusters using the 4B query vector, and searches
only their window rows using the 0.6B query vector.
