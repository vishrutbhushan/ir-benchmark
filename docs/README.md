# IR Benchmark Documentation

## Start here

- [Repository quickstart](../README.md)
- [Architecture and data flow](../ARCHITECTURE.md)
- [Shared configuration](../config/pipeline.json)
- [Neural pipeline](../neural/README.md)
- [Classical pipeline](../classical/README.md)
- [Evaluation](../evaluation/README.md)
- [Tracked inputs](../data_inputs/README.md)

## Recommended execution order

1. Run shared preparation and filtering.
2. Run classical indexing/retrieval.
3. Run neural stages through verification.
4. Export results and evaluation metrics.
5. Start the search UI and test one query.
6. Repeat the same commands to verify safe stage reuse.

The commands are identical for every corpus size. Use the same runners with
different input paths or resource limits when a bounded validation run is
needed; this changes data volume only, never the pipeline implementation.

## Runtime locations

Each component separates:

- `outputs/`: indexes, checkpoints, vectors, databases, and markers;
- `results/`: CSV exports, reports, and metrics;
- `logs/`: timestamped stage logs.

Runtime data is ignored by Git. Synthetic evaluation labels must be generated
with `evaluation/run_evaluation.py --synthetic-random`, include the seed and
timestamp, and must never be treated as human relevance judgments.
