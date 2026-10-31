# Neural retrieval

This component consumes the shared filtered ZIP and runs extraction,
overlapping neural windows, Qwen3 embeddings, weight configuration,
classification, routing, verification, search, and the web UI.

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
