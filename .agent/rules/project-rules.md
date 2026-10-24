# Project rules

- This is a course comparison of classic BM25/TF-IDF and neural dense retrieval on the same technical HTML corpus. Changes here cover the neural engine and shared corpus; preserve teammates' classic work.
- Hand-code dense cosine ranking with NumPy/PyTorch matrix operations. Hand-code classic indexing/scoring and final NDCG@5/MAP; no retrieval/vector libraries such as FAISS, Annoy, Chroma, Lucene or Rank-BM25.
- Use one repository .venv and requirements-technical-web-corpus.txt. Use pretrained models only for embeddings; keep model revision and input conventions reproducible.
- Apply repairs/exclusions to technical_web_corpus itself and its normal manifest. Do not require a separate teammate allowlist. Keep the original ZIP and local recovery backup.
- Process the whole retained corpus for production builds. Small diagnostic fixtures and labeled learning subsets are allowed.
- Keep neural preprocessing separate from classic normalization. Neural text preserves negation, case, punctuation, technical identifiers, versions and code whitespace. Classic rules: Porter stemming only, atomic versions, split/unsplit identifiers and structured dates.
- Structured records contain doc_id, url, title, headings, sections, body_text, code_text, dates and metadata. Dates are mentions unless publication provenance is established.
- Keep simple, named modules and one CLI, run_pipeline.py. Outputs/logs are hierarchical and ignored; analytical stages save charts and a dashboard.
- The summary hierarchy is removed. The baseline uses pure-zone windows and exact cosine. Learned four-zone scoring (title/headings/body/code) and class routing are separate extensions; do not claim completed builds or improvements without verified artifacts. Table text belongs to body in the simplified target.
- Learning tables are self-contained. Relevance labels are query/document judgments; classification labels currently map source technologies to 14 fixed domains. Never equate the two or falsely attribute labels to independent human review.
- No annotation burden goes to the user. Keep development queries separate from final evaluation, split zone data by query, and group duplicate document families before classification splits. Fit on training data only.
- Keep one neural reference document and concise README commands. Do not include personal system specs, local paths, setup notes or session narrative in shared files. Private history stays in ignored .agent/local/.
- Discussion questions do not authorize implementation. Fix verified defects within an authorized task; ask before replacing agreed architecture.
- Maintain current decisions here. Do not commit or push unless requested; publishing is handled by the user through their editor. Never force push or alter main without explicit authorization.


### Implementation authorization — 2026-10-23
* Implement the agreed fixed four-zone production retrieval and learned weights, both classifiers and validated category routing, corpus-quality fixes, final query/evaluation preparation and tests. Preserve a plain cosine baseline. Process the entire corrected corpus, keep source labels and relevance labels distinct, and report measured results honestly. Do not implement teammates' classic engine or fabricate its results. No new push/commit is requested.

### Pause and resume — 2026-10-23
- The user explicitly paused the current build. Do not restart it until requested. Preserve all model/index files and embedding checkpoints; resume completed work rather than rebuilding it. Confirm cache fingerprints before resuming and never silently reset progress.

### Publishing authorization — 2026-10-24
- The user requested one commit and a push of the current source changes. Exclude generated/incomplete outputs and keep clone instructions concise. This does not authorize restarting the paused build. Push only the current development branch, without force.
