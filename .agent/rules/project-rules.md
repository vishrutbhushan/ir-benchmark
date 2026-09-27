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

## 2. Text Preprocessing & Tokenization Rules

* **Stemming:** Use Porter Stemmer consistently. Do NOT mix lemmatization with stemming.
* **Code & Technical Identifiers:** Explicitly preserve and handle hyphens, underscores, and camelCase. Index both the unsplit identifier and split components (e.g., `react-router-dom` -> `react-router-dom`, `react`, `router`, `dom`).
* **Version Strings:** Atomic token handling for versions (e.g., `3.11`, `v2.4` must NOT be split on dots).
* **Date Normalization:** Dates must be parsed and stored in a structured field (e.g. ISO format + doc ID mapping), separated from free-text body tokens.
* **Stopwords:** Use standard English stopwords plus domain/boilerplate stop terms identified during Phase 0 audit.

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

