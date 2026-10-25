# Neural branch guidelines

Read [.agent/rules/project-rules.md](.agent/rules/project-rules.md).

- This branch owns only neural extraction, embeddings, learning, retrieval and UI.
- Collection, repairs, shared filtering and corpus ZIP export belong to `data`.
- Final benchmark queries and future comparative evaluation belong to `evaluation`.
- Never modify BM25 branches and never push code.
- Hand-code cosine ranking and learning metrics; no vector-search libraries.
- Keep numbered steps, one `requirements.txt` and one `run_pipeline.py`.
- Keep source labels in `data/`, small reports/charts in `results/`, large runtime artifacts in ignored `outputs/`.
- Preserve completed embeddings and checkpoints. Verify cache compatibility after changes.
- Keep CSVs compact; duplicate-family grouping stays internal.
- Update README and project rules when behavior or constraints change. Commit verified changes with a descriptive message.
