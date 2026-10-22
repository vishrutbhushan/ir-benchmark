# Project knowledge and history

## 2026-10-20 — Repository review and retrieval planning

User scope: owns neural retrieval; teammates own classic retrieval. First deliver fast independent retrieval and a minimal paginated search UI; evaluation and novelty follow. Minimize manual work; keep technical-corpus issues visible. Corpus simplification is acceptable when justified and applied consistently. Hard instructions are recorded in `.agent/rules/project-rules.md`.

Inspected both supplied PDFs in full using `pdftotext`, existing source/reports, labels, queries and ZIP manifest. Course coverage supports exact vector scoring, classic indexing, probabilistic models, evaluation, clustering and web-search architecture. Submission proposes ANN; current repository rules prohibit ANN/vector retrieval libraries. Use hand-coded exhaustive matrix ranking and document the deviation later.

Verified: 79,268 HTML/manifest entries, 11 categories, 76 technologies; ~19.94 GB uncompressed HTML. All paths use Windows separators; normalized paths match ZIP members. No duplicate manifest SHA-256 hashes; near-duplicates still need checking. Kafka has three pages and security 62. Existing audit/extraction reports are samples; no extracted corpus or structured JSONL exists in this checkout. Current query file has 25 queries/eight categories, inconsistent with the planned 20/six categories.

Hardware outside sandbox: RTX 5070 Ti **Laptop** GPU, 12,227 MiB VRAM, driver 617.42; 24 CPU cores; host RAM ~31.4 GiB; WSL RAM ~15.3 GiB, swap 4 GiB. GPU failures inside the agent sandbox do not imply broken WSL passthrough. No WSL/host setting was changed. CUDA model/tensor throughput remains untested.

Initially ran the temporary diagnostic hook `run_pipeline.py --phase planning --sample-per-tech 1` (subsequently removed; see correction below) in a temporary `/tmp/ir-plan-env` environment. Seed 42, 76 documents, 35.49 seconds before chart writing; 47 contain extracted code, 52 have at least one section at the parser's 2,000-character cap (239/998 sections at cap). Synthetic probes confirm navigation heading retention, code whitespace loss, nested paragraph duplication, section truncation and date mentions retained in body. These results describe current parser behavior, not full-corpus quality or retrieval effectiveness.

**Recommended defaults, pending implementation:** separate lexical/neural preprocessing after shared cleanup; title/heading-aware passages; BGE-base-en-v1.5 first, Qwen3-Embedding-0.6B challenger; exact cosine and maximum-passage document aggregation; SQLite metadata + NumPy embeddings; local bounded LRU caching; FastAPI/Jinja search UI. Hybrid RRF follows independent baselines; RAPTOR is optional later research. These recommendations are not claims of user approval or completed functionality.

Artifacts: [implementation plan](../outputs/neural/history/documentation_before_cleanup/docs/RETRIEVAL_PLAN.md), [inspection report](../outputs/planning/report.md), [machine snapshot](../outputs/planning/hardware.json). Initially added a diagnostic runner hook and missing dependencies; both production changes were subsequently removed after the scope clarification below. Existing extraction behavior was deliberately left unchanged so its defects remain reproducible. Supplied untracked PDFs and their Zone.Identifier sidecars are user files and were left unmodified/uncommitted.

Next implementation gate: establish the pinned GPU environment, repair path resolution and extraction, and freeze the shared corpus/schema before generating embeddings. Retrieval engines, model downloads, full extraction and search UI have not been implemented by this planning pass.

### Scope correction from the user

The user clarified: **neural only; do not start implementation**. Only small tests are permitted. They supplied the original assignment: dense embeddings of the same corpus ranked with cosine; 20 complex queries explicitly including negation, homonyms and temporal constraints; NDCG@5, MAP and comparative failure analysis. Build a decent baseline before optional novelty.

The temporary diagnostic integration with `run_pipeline.py` was removed; the runner, requirements and all existing `src/` files are unchanged from HEAD. The small diagnostic survives only as `outputs/planning/reproduce_inspection.py`, with its results. No retrieval/UI implementation was started. README now documents the standalone experiment, not a new pipeline phase. [Neural plan](../outputs/neural/history/documentation_before_cleanup/docs/RETRIEVAL_PLAN.md) was rewritten around the original assignment and device-specific GPU setup was documented at that time (subsequently removed from the shared project).

## 2026-10-20 — Single environment and full raw-corpus validation

The user authorized environment repair and validation/fixes for the **entire** raw corpus before preprocessing. Use one repository `.venv` and the existing requirements file. Neural preprocessing must be separate from classical preprocessing. Retrieval/UI implementation remains deferred. The earlier no-implementation scope now permits this environment/data foundation work.

Created `.venv` using the existing Python 3.14.4, after confirming an official compatible CUDA PyTorch wheel exists. Consolidated corpus, extraction and neural runtime dependencies into the existing requirements file and pinned resolved versions. Added the VS Code default interpreter setting. Temporary bootstrap tools are not alternative project environments. GPU verification results are recorded separately when the final install completes.

Added the `validate` phase through the master runner, with modular `src/corpus_validation/` code. Every ZIP entry was read for CRC; all 79,268 HTML documents were checked against recorded size/SHA-256 and materialized without changing HTML bytes. All 79,268 Windows-style paths are normalized in a derived manifest. One original filename is 256 UTF-8 bytes, beyond Linux's limit; it now has a stable short alias with source path/filename preserved. Original ZIP and original manifest remain unchanged. ZIP SHA-256: `c7b66becf55219ed3a00b8df93b3b06287ea0c23ad59097aa7cc3b19b4bd9653`.

Validation initially identified RFC `<pre>` wrappers as non-HTML/code; inspection corrected that heuristic and the **entire corpus was rerun**. Final integrity failures: **0**. Automated content triage: **68,185 eligible**, **5,674 excluded** (5,425 confidently non-English, 249 effectively empty), **5,409 review**. Exact visible-text candidates: 1,846 groups/4,666 documents, retained pending policy decisions because versioned pages can share text. The eligible candidates cover 75 technologies/all 11 categories: all three Kafka captures are redirect stubs, so there is no eligible Kafka document. This impacts the candidate query suite. The eligibility policy remains unfrozen; uncertain pages are not admitted automatically and no preprocessing starts yet.

Six focused tests pass for safe paths, long UTF-8 names, byte-preserving materialization, mismatch protection, troubleshooting pages and preformatted RFC prose. Full reports/charts/dashboard are in `outputs/corpus_validation/`; bulk manifests/per-document checks stay local and are reproducible. `src/neural/preprocessing/` and `outputs/neural/` are reserved independently of classical preprocessing; no neural implementation is present.

Revised model guidance: Qwen3-Embedding-0.6B is the main candidate; test the 4B model rather than dismissing it. The 12 GB GPU is not restricted to BGE-base. The earlier 384-token proposal was a budget for a 512-token encoder, not a universal rule. Answers are in `docs/NEURAL_RETRIEVAL_EXPLAINED.md`. Next: finish/verify the shared CUDA environment, then resolve/freeze the corpus eligibility policy before building neural preprocessing.

Shared-environment verification completed successfully: Python 3.14.4, PyTorch `2.11.0+cu128`, Transformers `5.20.2`; CUDA 12.8 detects the RTX 5070 Ti Laptop GPU with compute capability 12.0. CUDA matrix multiplication and a tiny random Qwen3 forward pass both passed. `pip check` reports no broken requirements. No pretrained model was downloaded, no corpus embedding created, no system package/WSL setting changed. Evidence: `outputs/environment/verification.json` and installed/resolved package inventories. The next data decision is the frozen eligibility policy, not another environment setup.

## 2026-10-22 — Full neural implementation authorization

User authorized full corpus repair, independent extraction/preprocessing, a concrete
retrieval choice, implementation/tests, saved results, master runner and visible UI.
Earlier planning-only restrictions are superseded in this scope. No sampled production
replacement; classic teammates' code and historical evidence remain intact.

Chosen implementation: pinned Qwen3-Embedding-0.6B, 512-total-token section passages,
64-token maximum overlap, title/heading context, durable SQLite + NumPy, hand-coded
exhaustive cosine and maximum-passage document ranking. Optional extractive spherical
cluster hierarchy is explicitly RAPTOR-inspired, not faithful RAPTOR. Local search UI
has filters and pagination. No lexical normalization in neural inputs, no vector service.

All 249 empty/error captures received bounded recovery attempts: 32 repaired, 217
unresolved. A redirect-target test caught JavaScript-expression prefix matching for
Kafka; rejected homepage attempts were saved, target handling corrected, all three
real documentation pages recovered, and full extraction rerun before embeddings.
Final full extraction accounts for 79,268 sources: 68,418 retained / 10,850 excluded,
11 categories / 76 technologies. Shared allowlist SHA-256:
`d37ad4b1449b86fcfe1136bd9fefddea09fe949b8f8176041322fc7825a3097d`.

Full passage preparation: 767,679 passages / 609,960 sections, 512 maximum total
model tokens, 1,024 dimensions, approximately 1.46 GiB FP16 vectors. Every section's
550,976,640 characters was covered, and every passage matched its source offsets.
The 18-test regression suite includes durable-window resume and invalid hierarchy
ancestor detection. UI fixtures passed desktop/mobile rendering, actual Next-page
interaction and a 390-pixel no-overflow check in isolated Chrome profiles.

Encoder optimization: pinned causal Qwen3 uses right-padding and gathers the final
real token. Real tokens cannot attend to future padding, so omitting its redundant
padding mask permits Flash SDPA. A real 128-passage diagnostic agrees with the
original masked path (minimum vector cosine 0.999794); BF16 paths differ slightly.
The full final index uses one consistent configuration and a new fingerprint.


### Completed full build and independent reload

All 767,679 vectors completed and passed finite/norm checks. The consistent full
embedding run took 131.6 minutes; the 12,223-node hierarchy
(12,013 / 205 / 5 nodes) took 4.2 minutes. All tree edges and ancestor
paths passed. Vector SHA-256: `aa21a6f6a113224c648926e610562f94fb301a94a1fc2befaa8037062edaff13`.

Independent-process verification compared GPU and NumPy cosine over ALL 767,679 rows
(maximum error 2.38e-07) and ran all 25 candidate queries
in both modes. Latest uncached p95: dense 88.88 ms; hierarchy
142.38 ms. These measure engineering behavior, not relevance.
A repeated full master command reused all completed artifacts and the same vector hash.

Real browser/API tests passed against the persisted full index. Compact ranked-ID
caching now fetches only visible snippets, avoiding 1,000 lookups for page one.
Startup warmup precedes server readiness; cached next-page HTTP time was
1.26 ms. The 19-test suite includes cache pagination ranks
and SQL fetch counts. Screenshots and measured reports are in `outputs/neural/`.
Formal judgments and NDCG@5/MAP remain pending as requested. No code was pushed.


### Manual commit consolidation — 2026-10-22

At the user's request, undo the five local commits after `4b24108` (`origin/main`) without deleting working files; the user will make one consolidated commit manually. Generated `outputs/` and `logs/` are now ignored in full, including reports and dashboards. Previously tracked generated artifacts are removed only from Git's index. Raw corpus/archive, the shared environment, caches, local agent state, secrets and Windows download sidecars are ignored; source, course PDFs, queries and shared editor settings remain eligible for versioning. Existing utility scripts are preserved under `scripts/check_environment.py` and `scripts/reproduce_inspection.py`; their old local copies and all expensive build artifacts remain on disk. Durable measurements and architecture history remain in this document and the README. No replacement commit or push is made for this task.


### Required hierarchy — 2026-10-22

The user chose a hierarchical/RAPTOR-style component as part of the main system. Hierarchical scoring is now the default in Python search, CLI, API and UI. The server requires a current hierarchy and does not silently fall back; pure dense cosine remains an explicit comparison baseline. The existing full-corpus hierarchy and passage embeddings are reused, with no encoder/chunking changes. Code remains source text embedded alongside section context, preserving whitespace and punctuation; this does not imply syntax-aware parsing or code execution. Relevance gains from the hierarchy remain unestablished.

Validation: all 19 regression tests passed, including default hierarchical scoring, explicit dense baseline and rejection of missing hierarchy. A real local HTTP server searched the existing 68,418-document / 767,679-passage index successfully with omitted mode (hierarchical) and explicit dense mode; the UI defaults to hierarchy. The temporary server was stopped. No embeddings were regenerated and no commit was created.


### Fixed-zone simplification review — 2026-10-22

The user requested simple, deterministic title/headings/body zones and consideration of code/table zones, separate zone storage, explicit weighted scoring and learning weights. The handout explicitly assigns learning weights and vector scoring in Manning Chapter 6. A read-only scan of every one of the 68,418 retained structured records found title 68,418 (includes manifest/URL fallbacks), headings 59,582, prose/list body 67,250, code 38,290 and tables 19,577. These are current extraction availability counts, not accuracy claims for a new HTML extractor. Charts, report and dashboard are saved under `outputs/neural/zone_audit/`. `docs/NEURAL_ZONE_DESIGN.md` records the proposed five-zone design, query-time weighted cosine, development-only weight tuning and a simplified centroid hierarchy for candidate generation. Current retrieval code, embeddings, hierarchy and allowlist were not replaced. No commit or push was made.


### AI-judged development CSV and course slides — 2026-10-22

Created 44 new development queries, four per corpus category, with 264 manually AI-adjudicated query/document pairs drawn from full-index dense/hierarchical pools, within-technology/lower-rank candidates, seeded unrelated controls and two curated primary documents. Coverage: 260 distinct documents, 41 intended query technologies and 56 document technologies. Labels: 136 relevant, 128 nonrelevant, with grades/confidence/reasons/exact local source excerpts. Two language/encoding-problem rows are excluded from tuning, leaving 262 usable pairs. All queries have positive and negative usable examples. All pairs for a query share a split: 33 train queries and 11 validation queries; no exact final query was reused. AI labels are development silver data, not human gold or final relevance evaluation.

`data/neural/development_training.csv` adds five true maximum-per-zone cosine features and presence flags. The run embedded every window of the selected current extracted documents (6,548 unique windows), without replacing production indexes or fitting weights. Reproduce with `run_pipeline.py --phase development-features --batch-size 32`. Data methodology is in `data/neural/README.md`; charts/report/dashboard are under `outputs/neural/development/`. Verified all 264 evidence quotes, source URLs, split consistency, numeric feature bounds and CSV hashes. All 19 existing regression tests pass. Production vector SHA-256 remains `aa21a6f6a113224c648926e610562f94fb301a94a1fc2befaa8037062edaff13`. Training CSV SHA-256: `e788654dc9d04f3b5ccc9df983b4f31d607ac28e96eee536666dd4be39bff70f`.

The requested Google Drive course folder is accessible through Windows interoperability despite no `/mnt/g` mount. Copied four relevant PDFs to ignored local outputs and visually inspected rendered image-only Text Mining Part 1 slides. PDF page 13 shows the Naive Bayes document-category posterior; pages 17-20 cover parameter estimation and smoothing. `docs/COURSE_CLASSIFICATION.md` records page references and distinguishes category classification, relevance judgment and zone-weight learning. No category classifier, learned weights, replacement index, commit or push was produced by this task.


### Shared class taxonomy and full-corpus CSV — 2026-10-22

The user selected both Naive Bayes and centroid/Rocchio classification and requested a defined class set and CSV based on the slides/book. Read `Category_Labels.txt` (Cloud, AI/ML, Prog Lang, DB, ML Ops, Communication/msg passing) and defined 14 primary source-domain classes covering all 76 technologies. Definitions and explicit technology mappings are versioned in `data/neural/classification/`; original category labels were preserved. Notable splits separate search engines from databases, messaging/RPC from distributed processing, and ML compute from MLOps. Vault and OpenSSL join security/identity. These are metadata-derived weak labels, not new independent content judgments.

`run_pipeline.py --phase classification-data` exports all 68,418 retained documents to `outputs/neural/classification/document_classes.csv`, with source references and label provenance. Both classifiers use the same rows and groups: train 54,726, validation 6,846, test 6,846. Canonical version URL families and identical extracted-content groups are joined before deterministic class-stratified allocation; all 14 classes occur in every split. No cross-class content conflicts were found. Charts, report, dashboard and full validation JSON are stored beside the CSV; execution logs remain under `logs/neural/`.

Two complete exports were byte-identical: SHA-256 `ee2a6a4f5e601c5ebc534969fb4dc519652a292ad026a21e58aadbc99b074447`. Verified exactly one row per retained document, all technology mappings, complete class coverage and no identified content/version-group leakage across splits. All 19 regression tests pass. The data follows the PPT's fixed-class supervised document/class setup. Neither classifier has been trained by this data-preparation task; no retrieval artifacts or relevance judgments were replaced, and no commit or push was made.


### Four-zone scope and assistant-owned annotation — 2026-10-22

The user wants no manual labeling/review work for either zone-weight learning or category classification; the assistant will provide needed judgments and resolve ambiguities. Avoid repeated labeling-provenance disclaimers in normal explanations while preserving accurate project records. The simplified target is four fixed zones: title, headings, body and code. To avoid table-specific preprocessing without discarding ordinary visible information, table text is retained by generic body extraction; remove the separate table zone and table reconstruction/header-label logic. Source-dependent classifier labels and relevance labels remain separate tasks; both classifiers share the prepared classification CSV. No production extraction, index or training CSV was migrated in this design-only update. The existing five-zone feature CSV is historical prototype data and must be refreshed for the new four-zone body contract before fitting weights. No commit or push was made.


### Summary-based hierarchy removed — 2026-10-22

After misreading a discussion question as an implementation request, deleted the extractive topic-tree implementation and obsolete clustering/tree tests. Removed the hierarchy build phase, artifact loading, ancestor-score blend, server requirement, UI selector and configuration fields. The working engine now uses exhaustive passage cosine only; class-based routing and four-zone learned scoring remain separate planned changes. Existing source extraction, chunking and encoder files were preserved so expensive embeddings stay reusable. Historical topic-tree artifacts remain unused on disk; prior verification evidence was copied to `outputs/neural/history/summary_hierarchy_verification/`, and BUILD_RESULTS now identifies its old hierarchy measurements as historical.

Validation: all 17 remaining regression tests pass. A complete `--phase all` reused extraction, all 767,679 passage vectors and source mappings, verified all 609,960 sections / 550,976,640 source characters and ran all 25 candidate queries in dense mode. Saved vector SHA-256 remains `aa21a6f6a113224c648926e610562f94fb301a94a1fc2befaa8037062edaff13`. Real GPU-backed HTTP checks passed default dense retrieval, filters, distinct cached page 2 results, UI removal and HTTP 400 rejection of the removed hierarchy mode. Removed CLI phase is rejected. The temporary server was stopped. Documentation/config updated; no commit or push made.


### Confirmed removal and organized learning data — 2026-10-22

The user has now explicitly authorized keeping the summary hierarchy completely
removed. Removed its obsolete generated tree/vector files; historical verification
reports remain as evidence. Current passage embeddings are preserved. No commit
or push was made; the user retains the manual consolidated-commit workflow.

Moved learning code into separate zone_weights and classification packages. Zone
CSV inputs now live in data/neural/zone_weights, with compact labels separate from
source evidence. Removed judge identity columns from learning inputs; original
judgment provenance remains in this history and the local archived source data.
Four development zones include visible table text in body, replacing the earlier
five-feature prototype. This does not migrate the production passage index.

Classification now exports a balanced starter dataset: 420 documents across 14
source-domain classes, 20 train/5 validation/5 test per class. It uses existing
metadata-derived domain labels, not fresh independent content adjudication. The
full source-label manifest remains available. Neither classifier nor zone weights
has been trained. Held-out performance must determine whether to expand training.

Validation: all 17 regression tests passed; all 420 classification rows were checked
for balance, grouped split consistency and deterministic regeneration. All 264
four-zone feature rows passed finite-score, missing-zone, label and query-split
checks; all 44 queries retain positive and negative usable examples. The development
run covered 6,505 unique zone windows. Production vector SHA-256 is unchanged.
Classification and development reports/charts/dashboards were regenerated.


### Cleanup, self-contained learning tables and canonical corpus — 2026-10-22

The user requested removal of dead code/redundant docs, denormalized learning tables,
and actual corrections to the main dataset instead of a companion allowlist. This
explicitly supersedes the earlier immutable working-HTML policy; the original ZIP
remains immutable. The requested midsem-report stash was skipped at the user's
follow-up instruction. No report/course PDF was changed or stashed.

Applied the accepted full-corpus audit to technical_web_corpus itself: 68,418 HTML
files and normal manifest records remain, exactly matching searchable documents.
28 accepted repairs replace bad captures at their normal paths. There were 32
successful recoveries overall, but three URL aliases and one non-English document
were excluded. Moved 10,850 excluded captures, superseded files and original metadata
to ignored outputs/corpus_backup for recovery. Verified every retained file's size
and SHA-256 plus exact directory/manifest coverage. Removed the separate neural
allowlist. Source locations in extraction metadata were updated while preserving
all text, document ordering, passage offsets and encoder inputs. Preparation and
validation now recognize the corrected dataset; a cold corrected-dataset extraction
also works without the original ZIP or historical audit files.

Consolidated learning inputs to one training.csv per task. Zone data repeats query
text/document details beside binary relevance, split and four cosine features;
classification includes complete clean text, class ID/name and split. Source-domain
label origins remain unchanged. Both classifiers share the same 420 rows; all
264 relevance judgments and 262 usability decisions are preserved. Earlier tables
and evidence remain in ignored history. Definitions/mapping remain reference data.
The zone table is a neural adaptation of the book's Boolean zone-score example;
the classification table uses the book's labeled document/class format.

Consolidated documentation to README, docs/DESIGN.md, WSL setup and this history;
kept AGENTS/project instructions. Removed nested READMEs, superseded plans/explanations,
the one-off sample inspection script, obsolete single-mode API/CLI/cache plumbing
and unused imports. Kept callable classic-preparation modules and corpus crawler.
Import-only model/chunk cleanup required a one-time cache-signature migration;
original cache metadata and exact source hashes are recorded under
outputs/neural/history/import_cleanup_cache_metadata. No vector data was regenerated.

Validation: 20 regression tests pass, including actual repair/pruning/recovery,
corruption/unlisted-file rejection and cold extraction without original audit files.
The full build reuses all 767,679 vectors, verifies all 609,960 sections and
550,976,640 source characters, and retrieves 100 unique documents for each of the
25 engineering queries. Full NumPy/GPU cosine and cache checks pass. GPU timings
were collected alongside development feature generation, so are not isolated speed
benchmarks. No commit, push or stash was made.

Final cleanup checks: both denormalized training tables pass full row/schema/label
checks; all 420 stored classification texts match their source records, and every
canonical source path/hash agrees with all 68,418 extraction records. Regenerating
the classification table is byte-identical (SHA-256
5d7801f09630b11766db413feb5edc26762748fdd1104e9b71fc41eeb839d74f).
The live GPU-backed HTTP check passed UI loading and distinct cached page 2 results
after mode removal; its temporary server was stopped.


### Shared documentation scope — 2026-10-22

The user clarified that WSL/GPU setup was specific to their device and does not
belong in this shared team repository. Removed the setup guide, its README links
and the standalone device diagnostic script. Historical hardware measurements
remain context for past runs, not team setup requirements. Application runtime
CPU/CUDA support and project dependency commands remain unchanged. No commit or push.

### Publish development branch — 2026-10-22

The user explicitly requested a new develop_vishrut branch from main and a push
of the current changes. This supersedes the earlier manual-commit/no-push request
for this operation. Source, self-contained learning tables and concise project docs
are included; generated outputs, raw corpus, models, indexes, logs and local secrets
remain ignored. Main is not pushed; no force push is authorized.
