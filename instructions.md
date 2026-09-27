Given you already have `technical_web_corpus/documents/html/<technology>/*.html` and `metadata/corpus_manifest.jsonl` from that script, here's the full pipeline from here to a defensible comparison, split by who does what.

## Phase 0 — Corpus Audit (before any indexing)

**Agent:**
* Write a sampling script: pull a random few hundred docs across all technology folders, extract `<body>` text length, and flag docs below a text-length threshold (likely JS-rendered/empty-body pages).
* Run near-duplicate detection across the full corpus (minhash/simhash on extracted text) and output a cluster report (which doc IDs are near-duplicates of which).
* Detect and flag: non-HTML content mislabeled as HTML, non-English pages (langdetect), error/redirect pages captured as 200 (e.g., "404", "page not found" boilerplate in title/body).
* Produce a single audit report: counts per exclusion category, with example doc IDs for each.

**Human:**
* Review the audit report and the flagged samples manually (spot-check maybe 30–50 across categories) to confirm the automated flags aren't over/under-firing.
* Decide and sign off on the final exclusion rules (e.g., "exclude if body text < 200 chars after boilerplate removal," "exclude non-English," "collapse duplicate clusters to one representative doc").
* Decide final corpus size target and whether any technology categories should be dropped/capped to balance the corpus.

## Phase 1 — Structured Extraction

**Agent:**
* Build the HTML → structured record parser: for each surviving doc, extract URL, title, heading hierarchy (h1/h2/h3 with nesting), body paragraphs, code blocks, lists/tables, anchor text, and any date-like strings (regex/date-parser pass).
* Apply exclusion rules from Phase 0 and write the cleaned corpus to a new structured format (e.g., one JSON per doc, or a single JSONL) with fields: `doc_id, url, title, headings[], sections[], body_text, code_text, dates[], metadata`.
* Use a boilerplate-removal library (trafilatura or readability-lxml) for the body-text extraction step.
* Produce a corpus statistics report (doc count, size, per-technology breakdown, avg doc length, how many dates/code blocks detected) after this pass.

**Human:**
* Spot-check a sample of structured records against their source HTML to confirm extraction quality (titles/headings not garbled, code blocks not swallowed into body text, etc.).
* Flag any systematic extraction bug back to the agent for a fix pass.

## Phase 2 — Tokenization & Normalization

**Agent:**
* Implement the tokenizer/normalizer as its own module (used by both engines' preprocessing, though the ranking math itself stays separate — see Phase 3/4):
  - Case folding, stopword removal (standard list + any doc-boilerplate terms found in Phase 0 audit)
  - Stemming (pick Porter, apply consistently — no lemmatization mixed in)
  - Explicit handling for hyphens/underscores/camelCase identifiers (e.g. index both the split and unsplit form of `react-router-dom`)
  - Explicit handling for version strings/numbers as atomic tokens (`3.11`, `v2.4` not split on `.`)
  - Normalize detected dates into a structured field separate from body tokens (e.g., ISO date + doc_id lookup), not just left as free text
* Apply this to every doc, producing per-zone token lists (title_tokens, heading_tokens, section_tokens, body_tokens, code_tokens) needed for later BM25F.

**Human:**
* Review tokenizer output on a sample, especially on technical identifiers and version strings, since this directly affects exact-terminology queries later. Approve or request tuning.

## Phase 3 — Classic IR Engine (hand-coded BM25 → BM25F)

**Agent:**
* Build the inverted index from scratch: per-term postings lists, with separate term-frequency counts per zone (title/heading/section/body/code), plus per-zone document length and average zone length across the corpus.
* Implement IDF computation over the corpus vocabulary, by hand.
* Implement plain BM25 scoring (flat, single combined field) first as a standalone function.
* Write a small toy-corpus validation test (5–10 hand-crafted docs) with expected scores computed by hand or cross-checked once against a reference library — used only for validation, not shipped as your submission code.
* Extend to BM25F: combine per-zone scores with configurable per-zone weights into one document score.
* Build the query-side pipeline: same tokenizer/normalizer applied to the query, looked up against the index, scored, ranked.
* Run this engine end-to-end on the full cleaned corpus for a placeholder/dummy query to confirm it executes at full scale without memory/performance issues.

**Human:**
* Review the toy-corpus validation output and confirm the hand computation matches.
* Decide the initial BM25F zone weights (or agree on a small grid to try) — this is a judgment call, not something to leave to the agent by default.

## Phase 4 — Neural IR Engine (hand-coded ranking, pretrained embeddings)

**Agent:**
* Select and load a pretrained embedding model (e.g., E5, BGE) via HuggingFace — no training.
* Decide/implement chunking: passage-level embeddings (aligned to the section-level structure from Phase 1) vs whole-document embeddings — implement whichever is agreed (see Human step below), plus aggregation logic if passage-level (max or mean pooling of passage scores up to document score).
* Batch-embed the full corpus on GPU (fp16, reasonable batch size for 12GB), store embeddings + doc/passage IDs on disk (numpy array or similar).
* Hand-code the ranking step: embed the query with the same model, compute cosine similarity directly (matrix multiply) against the full corpus embedding matrix, sort, return top-k — no FAISS/ANN library call, no `semantic_search()` helper.
* Run this engine end-to-end on the full corpus for a placeholder query to confirm correctness and timing at scale.

**Human:**
* Decide passage-level vs whole-document embeddings before the agent builds it — this is a real design fork, not a detail to defer.
* Spot-check a few query→top-result pairs manually for sanity (does the top result look topically related at all) before trusting the pipeline for the full query suite.

## Phase 5 — Query Suite Design

**Human (this one is mostly yours, not the agent's):**
* Write the 20 queries by hand, across your categories: exact terminology, paraphrasing, negation/exclusion, homonyms/ambiguous terms, temporal/version constraints, multi-constraint combinations.
* Tag each query with its category in a structured file (`query_id, query_text, category`) — needed later for the failure-analysis breakdown.
* Sanity-check that the corpus actually contains documents that could plausibly answer each query (no point testing a query with zero relevant docs in your corpus) — this requires knowing your corpus content, so it's a human judgment call, though the agent can help by keyword-searching the corpus first to confirm coverage.

**Agent:**
* Given the human-authored query list, do a quick coverage check per query (e.g., simple keyword grep across the corpus) and flag any query with suspiciously low potential coverage, for the human to reconsider or adjust.

## Phase 6 — Full-Corpus Retrieval Runs

**Agent:**
* Run both engines (BM25F and neural) on all 20 queries against the **full corpus**.
* For each query, output top-N per engine (generous N, e.g., 20–30) with scores.
* Union + deduplicate the two engines' top-N lists per query into the pooling set, and produce a blinded, shuffled annotation sheet per query (doc content + doc_id, with engine source stripped/hidden).

**Human:**
* No action yet — this phase is fully automatable given Phases 3–5 are done correctly.

## Phase 7 — Human Annotation (Gold Set Construction)

**Human (this is the core manual bottleneck, split across your team):**
* For each query, judge every pooled document as relevant / not relevant (binary, or 3-point if you prefer) — blind to which engine retrieved it.
* Assign 2 queries to be double-judged by two different people independently, to report inter-annotator agreement.
* Record judgments in a structured format (`query_id, doc_id, relevance_label, annotator_id`).

**Agent:**
* Compute inter-annotator agreement (e.g., Cohen's kappa) on the double-judged queries and report it.
* Merge all judgment sheets into a single gold-set file keyed by query_id + doc_id.

## Phase 8 — Evaluation

**Agent:**
* Implement NDCG@5 and MAP computation by hand (or using a well-known formula reference, but write the computation yourselves given your "core should be coded" constraint likely extends to eval too — worth confirming with the professor if library eval metrics are acceptable, since NDCG/MAP implementations are standard and low-risk to hand-verify).
* Run both engines' full ranked lists against the gold set, compute both metrics per query and aggregated.
* Break results down by query category (using the tags from Phase 5) into a summary table.
* If BM25F vs plain BM25 comparison is wanted, add plain BM25 as a third row using the same gold set (no extra annotation needed — same relevance judgments apply).

**Human:**
* Review the results table for anything implausible (e.g., a metric of exactly 0 or 1 across the board, which usually signals a bug) before trusting it for the report.

## Phase 9 — Failure Mode Analysis

**Human (primary), Agent (supporting):**
* **Agent**: for each query, generate a side-by-side diff of the two engines' top-5 with relevance labels overlaid, and flag the queries with the largest metric gap between engines — this narrows down which queries are worth manually inspecting.
* **Human**: for each flagged query, read the actual retrieved documents and the query, and write the mechanistic explanation for why one engine won/lost (not just "semantic vs lexical" — the specific token-overlap or embedding-space reason). This is inherently a human interpretive task; the agent can gather evidence but shouldn't write the diagnosis.
* **Human**: organize findings by query category into the report structure.

## Phase 10 — Write-up

**Human:**
* Write the report narrative, methodology description, and failure analysis prose (the parts requiring judgment and interpretation).

**Agent:**
* Generate all tables, charts (metric comparisons per category), and pull exact numbers/examples from the logged outputs to insert into the report draft.

---

One thing worth flagging now rather than later: whether "core must be self-coded" is meant to extend to the **evaluation metrics** (NDCG/MAP) as well as retrieval, since that's ambiguous in what you've told me the professor said — worth a quick clarifying question to them, since it changes Phase 8's agent/human split slightly but not materially.