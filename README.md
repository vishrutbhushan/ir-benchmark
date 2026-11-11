# IR Benchmark: Classical vs. Neural vs. Hybrid Retrieval

An end-to-end information retrieval benchmark comparing classical lexical search (BM25, BM25F), dense neural retrieval, unsupervised cluster routing, and Reciprocal Rank Fusion (RRF) across a 45,881-document technical engineering corpus.

---

## Subsystems & Architecture

The repository is modularized into dedicated pipeline subsystems:

- [**Data Collection & Corpus Generation (`data/`)**](data/README.md): Common Crawl WARC acquisition, segment extraction, HTML validation, and repair harness.
- [**Content Filtering (`filter/`)**](filter/README.md): Boilerplate stripping, quality filtering, near-duplicate pruning, and archive packaging.
- [**Classical Retrieval (`classical/`)**](classical/README.md): Multi-zone tokenization, inverted indexing, Flat BM25, and Hierarchical BM25F scoring.
- [**Neural Retrieval (`neural/`)**](neural/README.md): Sliding window segmentation, dense embeddings, Manning zone weighting, K-Means++ clustering, and soft multi-membership routing.
- [**Evaluation (`evaluation/`)**](evaluation/README.md): Cranfield query pool, dual-LLM consensus judgments (Nemotron + Gemma), and graded metrics computation.

---

## Setup & Prerequisites

### 1. Python Environment

Requires Python 3.11+:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Corpus Acquisition Options

All benchmark queries, consensus judgments, and zone calibration data are tracked in the repository. For the raw corpus, you have two options:

- **Option A (Use existing archive)**: Place `technical_web_corpus.zip` in `data/00_corpus/` (or directly in the repository root).
- **Option B (Crawl from scratch)**: Generate the corpus directly from Common Crawl using the data collection harness in `data/` (see [data/README.md](data/README.md)).

---

## Running the Pipeline

You can run the full pipeline end-to-end using the master runner, or execute each stage individually.

### Option 1: Master Runner (`run_full_pipeline.py`)

Executes all pipeline stages sequentially:

```bash
# Run all stages end-to-end (Filter -> Classical -> Neural -> Evaluation)
python run_full_pipeline.py

# Or run up to search engine readiness (skipping evaluation):
python run_full_pipeline.py --stage up-to-engine
```

You can also target specific stages:
```bash
python run_full_pipeline.py --stage filter       # Stage 1: Validation, filtering, archive
python run_full_pipeline.py --stage classical    # Stage 2: Inverted index, BM25 & BM25F
python run_full_pipeline.py --stage neural       # Stage 3: Windows, embeddings, clusters, routing
python run_full_pipeline.py --stage evaluation   # Stage 4: Run queries & compute metrics
```

---

### Option 2: Step-by-Step Execution

Each subsystem can be executed directly using its standalone CLI:

#### Step 0: Data Collection & Corpus Generation (Optional)
If crawling fresh WARC segments from Common Crawl rather than providing a pre-existing archive:
```bash
python data/run_collect.py
```
*See [data/README.md](data/README.md) for concurrency flags, subset parameters, and output details.*

#### Step 1: Content Filtering & Deduplication
Strips boilerplate, applies quality thresholds, and produces the filtered production corpus (`45,881` documents):
```bash
python filter/run_filter.py --step all
```
*See [filter/README.md](filter/README.md) for individual step options.*

#### Step 2: Classical Lexical Indexing (BM25 & BM25F)
Tokenizes documents across 4 zones (title, headings, code, body) and builds the inverted index:
```bash
python classical/run_pipeline.py --phase all
```
*See [classical/README.md](classical/README.md) for zone scoring and index compression details.*

#### Step 3: Neural Retrieval Pipeline
Extracts structured text into SQLite, computes dense embeddings, trains zone weights, and constructs K-Means++ cluster routing with soft multi-membership:
```bash
python neural/run_pipeline.py --step all
```
*See [neural/README.md](neural/README.md) for configuration parameters and vector cache settings.*

#### Step 4: Cranfield Evaluation & Metrics
Executes the 15 benchmark queries across all systems and computes graded metrics against consensus judgments:
```bash
# 1. Prepare query runs across all search paradigms
python evaluation/run_evaluation.py --prepare

# 2. Score runs against consensus judgments (NDCG@5/10, P@1/3/5/10, Recall, MRR)
python evaluation/run_evaluation.py --metrics --force
```
*See [evaluation/README.md](evaluation/README.md) for judgment schemas, pooling details, and inter-annotator agreement.*

---

## Interactive Search UI

Once indexes are built (Steps 1–3), launch the local web interface to test and compare real-time queries across BM25, BM25F, Dense Neural, Cluster-Routed, and Hybrid RRF:

```bash
python run_search_ui.py
```
Open **`http://localhost:8000`** in your browser.
