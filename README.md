# Technical IR Benchmark — Classical IR & BM25 Engine

A high-performance Information Retrieval system featuring a hand-coded, zero-dependency **Classical IR Engine** evaluated over a 77,565-document technical web documentation corpus.

This branch focuses on:
- **Okapi BM25** (Flat Document Baseline)
- **Zone-Weighted BM25F** (`title`, `heading`, `code`, `body`)
- **Hierarchical Multi-Level Indexing** (Level 1: Document Zonal Index, Level 2: Section / Passage Sub-Index, Level 0: Hierarchical Taxonomy Routing)
- **Tier-1 Champion Lists** for fast postings candidate pruning

> **Constraint & Design Rule**: All core indexing structures, posting lists, scoring equations (BM25 / BM25F), and evaluation metrics (NDCG@5, MAP, P@5) are hand-coded from scratch without external retrieval or search libraries (no Lucene, Rank-BM25, Whoosh, or FAISS).

---

## Quick Setup

Activate your virtual environment and install the required dependencies:

```bash
# Activate virtual environment
source .venv/bin/activate       # macOS / Linux
# or: .venv\Scripts\activate    # Windows

# Install required dependencies
pip install -r requirements-technical-web-corpus.txt
pip install trafilatura langdetect matplotlib pillow python-dateutil
```

---

## Pipeline Execution

All pipeline phases are run via the unified CLI runner `run_pipeline.py`.

### 1. Phase 1 — Structured Extraction
Parses raw crawled HTML files into structured records (`title`, `headings`, `sections`, boilerplate-free `body_text`, isolated `code_text`, and normalized `dates`):

```bash
# Run stratified extraction (small sample)
python run_pipeline.py --phase 1 --stratified --max-docs 500

# Run full corpus extraction (77,565 documents)
python run_pipeline.py --phase 1 --num-workers 8
```
- **Outputs**: `outputs/phase1/structured_records.jsonl`, `outputs/phase1/corpus_statistics.json`, `outputs/phase1/corpus_statistics.md`, `outputs/phase1/charts/`
- **Logs**: `logs/phase1/extraction.log`

---

### 2. Phase 2 — Tokenization & Linguistic Normalization
Processes extracted records with technical-aware tokenization (preserves version numbers like `3.11`, hyphenated identifiers like `react-router-dom`, code symbols, and Porter stemming):

```bash
# Run tokenization with 8 parallel worker processes
python run_pipeline.py --phase 2 --num-workers 8
```
- **Outputs**: `outputs/phase2/tokenized_corpus.jsonl`, `outputs/phase2/tokenization_stats.json`, `outputs/phase2/tokenization_stats.md`, `outputs/phase2/charts/`
- **Logs**: `logs/phase2/tokenization.log`

---

### 3. Phase 3 — Classical IR Engine (Multi-Level Indexing, BM25 & BM25F)
Constructs the hand-coded multi-zone inverted index, section sub-index (851,580 passages), category directory, and evaluates the benchmark query suite:

```bash
# Run both Flat BM25 and Hierarchical BM25F side-by-side (default)
python run_pipeline.py --phase 3 --scorer both

# Run with Category-First Hierarchical Routing (Level 0 Taxonomy Sub-Index)
python run_pipeline.py --phase 3 --scorer bm25f --routing-mode hierarchical_first

# Run with Tier-1 Champion List candidate pruning (Acceleration)
python run_pipeline.py --phase 3 --scorer bm25f --routing-mode tiered

# Run with custom BM25 hyperparameters
python run_pipeline.py --phase 3 --k1 1.5 --b 0.8 --top-k 30
```
- **Outputs**:
  - `outputs/phase3/inverted_index.json` (Inverted postings with zonal counts & positions)
  - `outputs/phase3/document_catalog.json` (Catalog with lengths and section catalog)
  - `outputs/phase3/bm25_results.json` (Flat Okapi BM25 rankings)
  - `outputs/phase3/bm25f_results.json` (Zonal BM25F rankings)
  - `outputs/phase3/multi_level_results.json` (Document + Best Section passage rankings)
  - `outputs/phase3/phase3_summary.md` (Performance & NDCG@5 comparison report)
  - `outputs/phase3/charts/` (Visual comparison plots)
- **Logs**: `logs/phase3/classic_ir.log`

---

### 4. Run Entire End-to-End Pipeline
```bash
python run_pipeline.py --phase all
```
* **Interactive Dashboard**: Open `outputs/dashboard.html` in your browser for the full interactive visual report across all stages.

---waht 

## Architectural Overview

```
                      Query String
                           │
                 [query_processor.py]
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
 [bm25_scorer.py]   [bm25f_scorer.py]   [multilevel_index.py]
   (Okapi Flat)       (Field-Weighted)    (Doc + Passage + Champion)
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                  [inverted_index.py]
             (Postings, Zonal Freqs, Catalog)
                           │
                           ▼
                [classic_ir_runner.py]
       (Executes Benchmark, Computes NDCG@5 & MAP,
               Outputs Results & Charts)
```

### Retrieval Models
| Model | Granularity / Zones | Formula / Strategy |
| :--- | :--- | :--- |
| **Flat BM25** | Whole Document | Okapi BM25 ($k_1=1.2, b=0.75$) over all terms |
| **BM25F** | 4 Zones: Title ($w=4.0$), Heading ($w=2.5$), Code ($w=2.0$), Body ($w=1.0$) | Weighted pseudo-frequencies $\tilde{f}(t, D)$ with length normalization per zone |
| **Multi-Level** | Level 1: Document + Level 2: Section/Passage | Fuses document relevance with highest scoring passage snippet: $\text{Score} = \alpha \cdot \text{BM25F}_{\text{doc}} + (1 - \alpha) \cdot \max(\text{BM25}_{\text{sec}})$ |
---

## Taxonomy & Categories

The corpus documents and search queries are structured using a **2-tier hierarchical ontology** defined in [`Category_Labels.txt`](file:///Users/b.ajay/Downloads/ir-benchmark/Category_Labels.txt) and mapped in [`src/utils/taxonomy.py`](file:///Users/b.ajay/Downloads/ir-benchmark/src/utils/taxonomy.py).

The 77,565 documents are partitioned across **13 canonical technical categories** and 5 high-level domains:

| # | Category | Domain | Sample Technologies (76 Total) |
| :--- | :--- | :--- | :--- |
| 1 | **Cloud** | Infrastructure | AWS, Azure, Google Cloud, Cloudflare, DigitalOcean |
| 2 | **AI/ML** | AI & Machine Learning | PyTorch, TensorFlow, Hugging Face, CUDA, Ray |
| 3 | **Prog Lang** | Software Engineering | Python, Java, Go, Rust, TypeScript, JavaScript, Kotlin |
| 4 | **DB** | Data Systems | PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, ClickHouse |
| 5 | **ML Ops** | AI & Machine Learning | Kubeflow, MLflow |
| 6 | **Communication/msg passing** | Data Systems | Kafka, Pulsar, RabbitMQ, ZooKeeper, gRPC |
| 7 | **Containers & Orchestration** | Infrastructure | Docker, Kubernetes, Podman, Helm, Nomad |
| 8 | **Web Dev** | Software Engineering | React, Vue, Next.js, Django, FastAPI, Tailwind CSS |
| 9 | **Observability** | Operations & Reliability | Prometheus, Grafana, OpenTelemetry, Jaeger, Datadog |
| 10 | **Testing** | Software Engineering | Pytest, JUnit, Playwright, Cypress, Selenium |
| 11 | **Version Control & CI/CD** | Operations & Reliability | Git, GitHub Actions, GitLab CI, ArgoCD, Jenkins |
| 12 | **Data & Analytics** | Data Systems | Spark, Flink, dbt, Airflow, Snowflake, Presto |
| 13 | **Security & Identity** | Operations & Reliability | HashiCorp Vault, OWASP, NIST, OpenID Connect |

### How Categories are Used in Retrieval
1. **Document-Level Category Postings**: Inverted postings are pre-partitioned into `category_postings[category]` so sub-indices can be queried independently without scanning the entire corpus.
2. **Query Taxonomy Inference**: When a user queries (e.g., *"how to fix cuda out of memory in pytorch"*), `infer_query_taxonomy()` uses word-boundary regex matching across technology aliases to detect target categories (`AI/ML`).
3. **Category-First Routing & Soft Boost**:
   - In `routing_mode="hierarchical_first"`, the engine queries the category sub-index first.
   - In `routing_mode="hybrid"`, documents matching the query's inferred category receive a principled relevance boost ($\times (1 + \text{hierarchical\_boost})$).

---

## Directory Structure

```text
ir-benchmark/
├── run_pipeline.py               # Master CLI runner
├── queries.jsonl                 # 25 benchmark queries across technical domains
├── Category_Labels.txt           # Master taxonomy labels
├── src/
│   ├── phase1_extraction/        # HTML boilerplate stripping, code isolation, structure
│   │   ├── boilerplate.py        # Chrome removal and <pre>/<code> extraction
│   │   ├── html_parser.py        # Structural parsing: headings, sections, dates
│   │   ├── extraction_runner.py  # Parallel multi-core extractor
│   │   └── corpus_statistics.py  # Quality validation and distribution metrics
│   ├── phase2_tokenization/      # Technical-aware tokenizer & normalizer
│   │   ├── tokenizer.py          # Porter stemmer, code tokens, version strings
│   │   ├── tokenization_runner.py# Parallel batch tokenization
│   │   └── test_tokenizer.py     # Tokenizer test suite
│   ├── phase3_classic_ir/        # Hand-coded Classical IR Engine
│   │   ├── inverted_index.py     # Postings, positional indices, and catalog storage
│   │   ├── bm25_scorer.py        # Hand-coded Okapi BM25 scorer
│   │   ├── bm25f_scorer.py       # Hand-coded Zone-Weighted BM25F scorer
│   │   ├── multilevel_index.py   # Multi-Level Document & Passage retrieval
│   │   ├── query_processor.py    # Query tokenizer and taxonomy router
│   │   ├── classic_ir_runner.py  # CLI benchmark evaluation & charts
│   │   └── test_classic_ir.py    # Complete IR verification test suite
│   └── utils/                    # Common taxonomy, charting, and logging utilities
│       ├── taxonomy.py           # 76-technology hierarchical taxonomy ontology
│       ├── chart_visualizer.py   # Matplotlib automated chart generation
│       ├── html_dashboard.py     # Interactive dashboard generator
│       └── logger.py             # Structured multi-phase logging
├── outputs/                      # Generated results, JSON reports, charts, and dashboard
└── logs/                         # Execution logs per phase
```
