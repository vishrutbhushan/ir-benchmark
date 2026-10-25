# Neural project rules

- Branch ownership: `data` owns collection, archive validation, repairs, shared filtering and corrected ZIP export; this branch owns neural retrieval; `evaluation` owns final benchmark queries and future evaluation. Never modify BM25 branches. Never push.
- Consume the corrected ZIP/folder from `data` and preserve original document IDs. Neural extraction may verify input integrity but must not silently exclude or repair corpus documents.
- Process the full retained corpus. Preserve completed vectors and verify cache fingerprints and content before migrating artifacts. Do not restart embedding from zero for a layout-only change.
- Use pinned Qwen3 embeddings and hand-coded cosine matrix operations. No FAISS, Annoy, Chroma or other delegated retrieval ranking.
- Neural inputs preserve negation, stopwords, case, versions, punctuation, identifiers and code indentation. Tables join body text; no language-specific code parsing or execution.
- Use four fixed zones: title, headings, body and code. The summary/RAPTOR hierarchy is removed. Keep plain dense retrieval separate from learned zone and class-routing variants.
- Fit on training examples only. Keep development queries in learning tables; final benchmark queries live on `evaluation`. Source-domain class labels are not relevance judgments. Do not falsely attribute labels to independent human review.
- Classification CSV fields are doc_id, class_id, class_name and split. Resolve full text by ID from SQLite. Group duplicates internally before splits; do not expose group_id in the CSV.
- Store readable source references in each numbered learning-data folder's documents/ directory, one file per document ID.
- Use numbered source steps, one requirements.txt, one run_pipeline.py and one environment. data/ holds inputs, results/ holds versioned small charts/metrics/JSON, outputs/ holds ignored models, vectors and working databases. ZIPs and embeddings must never enter Git.
- Keep README concise and accurate. Keep one neural architecture reference; no PDFs, course-text dumps, device specifications or session history in shared files. Private history stays under .agent/local/.
- Questions do not authorize implementation. Complete authorized fixes and commit verified source/data/results changes locally; never push.

- Current reorganization publishing authorization: user permits pushes and will complete GitHub login if needed. Publish only source, compact label/configuration tables and small analytical results. Never publish corpus/source-document folders, ZIPs, SQLite databases, model weights or embeddings; inspect outgoing commit history as well as the final tree. Do not modify BM25 branches.

- Ignore legacy corpus folders and large binary artifacts as well as the numbered runtime folders. Keep large local inputs/artifacts out of editor file watching; distinguish actual Git changes from stale editor status.
