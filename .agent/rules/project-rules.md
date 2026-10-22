# IR Benchmark Project Rules & Hard Instructions

This document is the single source of truth for hard project constraints, architectural requirements, and rules for all coding agents working on the `ir-benchmark` repository.

---

## 1. Core Architectural & Implementation Constraints

### Hand-Coded Core Policy
* **No Off-the-Shelf Retrieval Libraries for Core Engines:**
  - **Classic IR (Phase 3):**
    - Hand-code inverted index from scratch (postings lists, zone frequencies, zone lengths).
    - Hand-code IDF computation across corpus vocabulary.
    - Hand-code BM25 (flat) and BM25F (zone-weighted) scoring functions.
    - Do NOT use Lucene, Whoosh, Rank-BM25, or Elasticsearch for the benchmark retrieval.
  - **Neural IR (Phase 4):**
    - Use pretrained HuggingFace models (e.g., BGE, E5) only for feature extraction (embeddings). No training required.
    - Hand-code cosine similarity and ranking via direct matrix multiplication (e.g., PyTorch/NumPy).
    - Do NOT use FAISS, Annoy, Chroma, Pinecone, or `sentence_transformers.util.semantic_search()`.
  - **Evaluation Metrics (Phase 8):**
    - Hand-code NDCG@5 and MAP calculation logic to ensure complete algorithmic transparency and defensibility.

---

## 2. Classical Text Preprocessing & Tokenization Rules

* **Stemming:** Use Porter Stemmer consistently. Do NOT mix lemmatization with stemming.
* **Code & Technical Identifiers:** Explicitly preserve and handle hyphens, underscores, and camelCase. Index both the unsplit identifier and split components (e.g., `react-router-dom` -> `react-router-dom`, `react`, `router`, `dom`).
* **Version Strings:** Atomic token handling for versions (e.g., `3.11`, `v2.4` must NOT be split on dots).
* **Date Normalization:** Dates must be parsed and stored in a structured field (e.g. ISO format + doc ID mapping), separated from free-text body tokens.
* **Stopwords:** Use standard English stopwords plus domain/boilerplate stop terms identified during Phase 0 audit.

These lexical token transformations do not apply to neural model inputs. The neural
pipeline preserves natural text, stopwords, negation, case, technical punctuation,
code and versions, and uses the selected model's own tokenizer and query formatting.

---

## 3. Data Ingestion & Extraction Standards

* **Phase 0 Corpus Audit Requirements:**
  - Empty/JS-rendered body filtering based on text length threshold.
  - Near-duplicate detection using MinHash / SimHash clustering.
  - Language filtering (English only via `langdetect` or fasttext).
  - Error/redirect page detection (e.g. 404/page not found boilerplate).
* **Structured Record Format:**
  Every processed document must conform to:
  `doc_id, url, title, headings[], sections[], body_text, code_text, dates[], metadata`
* **Boilerplate Removal:** Use `trafilatura` or `readability-lxml`.

---

## 4. Evaluation & Benchmarking Protocol

* **Query Suite:** 20 queries across 6 categories (Exact terminology, Paraphrasing, Negation/exclusion, Homonyms/ambiguity, Temporal/version constraints, Multi-constraint).
* **Pooling:** Top-N (20–30) per engine unioned and deduplicated.
* **Blinded Annotation:** Annotator pools must have engine provenance stripped and doc order shuffled.
* **Agreement:** Compute Cohen's kappa on double-judged subsets.
* **Breakdown:** Results must be reported overall and broken down by query category.

---

## 5. User-Appended Hard Instructions (Living Record)

*Any specific constraints, decisions, or rules provided by the user in future conversations must be appended below so subsequent agents never lose context.*

* (Baseline) All instructions from `instructions.md` are adopted as baseline project rules.
* **Code Organization & Modularity**: Proper hierarchical folders and clean, well-named modular files. No monolithic big files doing everything.
* **Outputs & Logs Hierarchy**:
  - All analytical and artifact outputs must be saved to a structured, hierarchical `outputs/` directory (e.g., `outputs/phase0/`, `outputs/phase1/`, etc.).
  - All execution logs must be saved to a structured, hierarchical `logs/` directory (e.g., `logs/phase0/`, `logs/phase1/`, etc.).
* **Single Runner Script**: All pipeline phases and steps must be executable and triggered via a single master script (e.g., `python run_pipeline.py --phase <phase> ...`) that sets up hierarchical outputs and logging.
* **README Maintenance**: Every time, `README.md` must be kept up to date with changes. The README should be simple, highlighting the overall idea and exact commands to run, rather than low-level implementation details.
* **Git Commit on Every Change**: Every time changes are made, create a descriptive git commit.
* **STRICT RULE — NEVER PUSH CODE**: Never run `git push` or push code to any remote repository under any circumstances.
* **Visualized Outputs**: All phase analytical outputs must be visualized with charts and graphs (e.g., category distributions, document length distributions, exclusion breakdowns, code block/date prevalence) saved hierarchically in `outputs/<phase>/charts/`, along with visual markdown figures and interactive dashboards.
* **Phase 0 Deep Analytics & Equal Representation**: Phase 0 audit must employ balanced, stratified sampling ensuring equal representation across all 11 technical categories and 76 technologies. Analytical insights must comprehensively cover: language breakdown (multilingual distributions & source tech hosts), code vs no-code prevalence across categories, DOM complexity (headings, tables, lists), HTML boilerplate overhead ratio, and duplicate clustering dynamics. Every analytical dimension must be visualized with dedicated charts and integrated into the interactive dashboard.


### Planning request — 2026-10-20
* The user's responsibility is neural retrieval; teammates own classic retrieval. Plan shared interfaces without taking over their implementation.
* Inspect the submitted project PDF, course handout, existing code, corpus ZIP, category labels, queries, and actual WSL hardware before recommending architecture.
* Prioritize fast, working classic and neural retrieval plus a simple search UI with pagination. Formal evaluation, enhancements, and novelty come later, but preserve evaluation compatibility now.
* Compare RAPTOR, section/title hierarchy, category-based grouping, exact cosine ranking, hybrid retrieval, model options, preprocessing timing, persistence, and caching with concise advantages and disadvantages.
* Distinguish lexical normalization from model-native neural preprocessing; explain treatment of technical identifiers, versions, dates, and code snippets.
* Minimize manual work and support coding-agent implementation. Small-scale diagnostic tests are authorized; corpus simplification is acceptable if documented and justified.
* Maintain project knowledge/history and produce a concise, actionable plan. Hardware/WSL settings should be inspected and recommendations documented; no host settings changes are requested.

### Scope clarification — 2026-10-20
* Plan ONLY the user's neural retrieval responsibilities. Classic implementation belongs to teammates; discuss only the integration contract needed for the comparison.
* Do NOT start actual implementation. Only small diagnostic tests are authorized now. Keep retrieval/UI implementation and system-settings changes as documented future steps.
* The original assignment requires an open-source embedding model or embedding API, the same corpus as classic retrieval, cosine ranking, 20 complex queries explicitly covering negation, homonyms and temporal constraints, NDCG@5, MAP and comparative failure analysis.
* Provide GPU/WSL setup steps the user can follow; do not change host settings. Distinguish agent sandbox GPU restrictions from actual WSL GPU access.

### Environment and corpus-first instructions — 2026-10-20
* Use ONE shared repository Python environment (`.venv`) and the existing requirements file as the dependency entry point; do not create a separate neural environment.
* Environment setup and whole-corpus integrity validation/repair are now authorized. Validate every corpus document, not only samples, before neural preprocessing or embedding.
* Keep raw HTML immutable; distinguish archive/file integrity from content suitability and extraction defects. Do not claim all HTML is correct based on sample checks.
* Neural preprocessing must be independent of classical preprocessing, with separate folders/pipeline stages while retaining the master CLI entry point and a shared validated raw-corpus contract.
* Answer the user's conceptual questions about stronger models, RAPTOR, dense retrieval, database use, embedding memory, and token/chunk sizing. Retrieval/UI implementation remains deferred.

### Full implementation authorization — 2026-10-22
* Implement the corpus fixes, repair/download incomplete or missing source captures where needed, and implement the independent neural extraction/preprocessing and runnable retrieval path. This supersedes earlier planning-only restrictions within this scope.
* Choose one concrete retrieval/index architecture and implement/test it; do not leave a menu of unresolved alternatives.
* Process the ENTIRE corpus/retained dataset, not a sampled replacement. Small unit tests are allowed for correctness, but final runs and verification must cover full data.
* Save important artifacts/results, follow the `run_pipeline.py` structure, maintain knowledge/README, and create local descriptive commits. Never push.
* Remove genuinely dead/irrelevant code within the changed scope; preserve teammates' classic implementation and useful historical evidence.
* Provide exact runnable commands and a visible working result. Continue using the single `.venv` and existing requirements file.

### Git consolidation request — 2026-10-22
* Undo the five local commits while preserving all working files and generated data. The user will create one consolidated commit manually; do not create a replacement commit for this task. This overrides the automatic-commit rule for this consolidation.
* Ignore generated `outputs/`, logs, raw corpus/archive, environments, caches and local machine artifacts. Remove previously tracked generated artifacts from Git's index without deleting local files. Preserve reusable scripts outside generated-output folders.

### Hierarchy architecture decision — 2026-10-22
* Include the hierarchical/RAPTOR-style component as part of the main neural system, not an optional enhancement. Make hierarchical scoring the default retrieval path; keep plain passage cosine as an explicit experimental baseline for the course comparison.
* Preserve the distinction between the implemented clustering/extractive hierarchy and faithful RAPTOR. Code snippets are embedded as source text with surrounding section context; no code execution or syntax-tree encoder is implied.

### Simplicity and deterministic zones — 2026-10-22
* Prefer a simple, course-grounded neural design with fixed, deterministic document zones, rather than growing format-specific extraction rules. The user accepts some content loss in exchange for predictable sections.
* Assess title, headings and body as the core zones, with code and tables as possible additional zones. Explain separate zone storage/indexing, query-time weighted scoring, how weights can be selected or learned, and relevant textbook/course concepts before restructuring the existing index.
* Check which zone definitions work across the full corpus. Preserve the user's preference for a hierarchical component while simplifying its role; do not treat proposed zone weights as learned or validated without labeled evidence.

### Development judgments and course slides — 2026-10-22
* Create a CSV containing a reasonably sized, diverse subset of actual corpus query/document pairs, with the assistant judging relevant/nonrelevant for zone-weight development. Identify the judgments as AI-generated, preserve source evidence and keep development queries separate from final evaluation queries.
* Check access to `G:\My Drive\BITS Documents\Information Retrieval` for class slides. Investigate textbook category creation and probabilistic document classification, and explain whether they are appropriate for this neural retrieval system.

### Shared classification data for both methods — 2026-10-22
* The user wants both Naive Bayes and centroid/Rocchio-style classification considered, using the course slides/book methodology. First define an explicit class set from the existing category labels, assess sensible additions, and create a document-class CSV usable by both methods.
* Distinguish class labels from query/document relevance labels. Reuse existing labels where appropriate instead of repeating relevance annotation. Any expanded classes must have clear definitions and support in actual corpus documents; disclose metadata-derived labels and any remaining validation needs.

### Minimal annotation effort and simpler preprocessing — 2026-10-22
* The user does not want to perform manual annotation or review for zone-weight learning or category classification. The assistant must supply needed judgments and resolve ambiguous cases rather than handing annotation work back to the user.
* The user requests no repeated AI-label disclaimers in normal explanations; keep accurate provenance in project records without falsely attributing labels to humans or claiming independent validation.
* Keep a small fixed zone schema and avoid unnecessary table/format preprocessing for this course project. The simplified next-design recommendation is title, headings, body and code. Remove the separate table zone; retain visible table text through generic body-text extraction, without table reconstruction, column-label expansion or table embeddings. This is a revised design scope, not a claim that the existing five-zone prototype CSV or production extraction has already been migrated.

### Remove summary-based hierarchy — 2026-10-22
* Remove the existing extractive-summary hierarchy and its ancestor-score blending from the active implementation, pipeline and UI. This supersedes the earlier requirement that this specific hierarchy be mandatory/default.
* Preserve the reusable corpus, passage embeddings and existing local build evidence. Use plain cosine retrieval as the working path until the separately planned class-based routing/four-zone design is implemented; do not claim that replacement is already working.

### Confirmed removal and simple learning datasets — 2026-10-22
* The earlier question “can we remove” was discussion, not authorization. The user has now explicitly confirmed: keep the summary hierarchy completely removed; reconsider it later only if requested.
* Keep simple, properly named modular files/folders. Separate zone-weight and document-classification CSV folders, with compact textbook-style supervised input tables and source evidence separate from training features. Do not add author/judge disclosures to training CSVs or falsely attribute labels to humans.
* Small labeled learning subsets are allowed for both tasks; the production corpus/index still covers every retained document. No annotation burden is assigned to the user. Keep training/validation/test separation and document actual label origins in project history.
* Preserve the user's manual consolidated-commit workflow; do not commit or push.

### Cleanup and canonical shared dataset — 2026-10-22
* Remove genuinely unused/obsolete code and redundant documentation; retain essential instructions, concise run commands and project history. Denormalize learning CSVs into self-contained textbook-style training tables where practical.
* Stash only midsem-report changes once its actual path is identified; do not stash unrelated implementation, course inputs or other PDFs.
* Apply accepted repairs and exclusions to technical_web_corpus itself, updating its normal manifest and actual HTML files. Teammates must consume the corrected dataset directly, without a separate shared allowlist. This supersedes keeping the extracted working HTML immutable. Keep the original ZIP and a local recovery backup intact.
* Follow-up: the user said to skip the midsem-report request. Do not stash or change report/course input files.

### Shared team repository, not personal device setup — 2026-10-22
* Remove personal WSL/GPU setup guides and diagnostic-only scripts from shared project documentation/code. The user's device investigation was local context, not setup instructions or hardware requirements for teammates. Keep shared documentation focused on the application, dataset and reproducible project commands; retain historical measurements only as past-run evidence.

### Explicit branch and push authorization — 2026-10-22
* The user explicitly requests creating develop_vishrut from main and pushing the current changes. This authorizes one consolidated commit and a normal push of that branch, superseding the earlier manual-commit/no-push restrictions for this operation. Do not push main or use force push.
