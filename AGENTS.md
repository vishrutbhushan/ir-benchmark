# Agent Instructions & Project Guidelines

This file is automatically loaded by AI coding agents.

The detailed rules and hard constraints are maintained in:
[.agent/rules/project-rules.md](file:///c:/Code/ir-benchmark/.agent/rules/project-rules.md)

### Key Rules at a Glance:
1. **Hand-Coded Core**: BM25 / BM25F indexing & scoring, dense ranking (matrix multiplication cosine similarity), and evaluation metrics (NDCG@5, MAP) must be hand-coded from scratch (no off-the-shelf retrieval/vector libraries like Lucene, Rank-BM25, FAISS, or Annoy).
2. **Tokenizer Rules**: Porter stemming only; atomic version numbers (`3.11`, `v2.4`); preserve both split and unsplit technical identifiers (`react-router-dom`); separate structured dates.
3. **Structured Format**: `doc_id, url, title, headings[], sections[], body_text, code_text, dates[], metadata`.
4. **Modularity & Hierarchy**: Clean, modular files. Hierarchical `outputs/` and `logs/`. Master single CLI runner (`run_pipeline.py`).
5. **README Maintenance**: Keep `README.md` up to date with changes every time. Keep it simple, focusing on the overall idea and exact commands to run.
6. **Git Commits**: Create a descriptive git commit every time changes are made.
7. **NEVER PUSH CODE**: Never execute `git push` or push code to any remote repository.
8. **Agent Memory**: Whenever the user provides hard instructions, design choices, or constraints, immediately append them to [.agent/rules/project-rules.md](file:///c:/Code/ir-benchmark/.agent/rules/project-rules.md).
