# IR Benchmark: Classical BM25 vs. Dense Neural vs. Hybrid RRF

Comprehensive information retrieval benchmark evaluating classical lexical matching (BM25, BM25F), dense neural retrieval (Qwen3-Embedding-0.6B), unsupervised cluster routing (K-Means++ with soft multi-membership), and Reciprocal Rank Fusion (RRF) across a 45,881-document technical engineering corpus.

---

## Benchmark Results (All 15 Evaluated Queries)

Evaluated under the **Cranfield paradigm** with pooled multi-annotator judgments (Nemotron-3.5-Bolt & Gemma 4:12b) and non-linear graded relevance discounting ($2^g - 1$ DCG):

| Method / Paradigm | Graded NDCG@5 | Graded NDCG@10 | P@5 | P@10 | Recall@10 | MRR@10 | Binary NDCG@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flat BM25 (Lucene Default)** | 0.7212 | 0.8600 | 0.7867 | 0.7000 | 0.5176 | 0.9333 | 0.9255 | 31.0 ms |
| **Hierarchical BM25F (Fields)** | 0.7406 | 0.8748 | 0.7333 | 0.6800 | 0.5010 | 0.9556 | 0.9237 | 43.5 ms |
| **Qwen3 Dense (Exhaustive)** | 0.7252 | 0.8639 | 0.8133 | 0.7333 | 0.5214 | 0.9222 | 0.9253 | 53.7 ms |
| **Cluster-Routed Neural (Soft)** | **0.8042** | 0.9088 | **0.8667** | 0.6867 | 0.4987 | 0.9667 | 0.9446 | **23.3 ms** |
| **Hybrid RRF (BM25F + Routed)** | 0.7690 | **0.9104** | 0.8133 | **0.7333** | **0.5380** | **1.0000** | **0.9547** | 66.8 ms |

* **Cluster-Routed Neural (Soft $\Delta \le 0.03$)**: Delivers the fastest neural query latency (23.3 ms, 2.3× speedup vs exhaustive scan) while achieving 0.9088 Graded NDCG@10 by indexing 12,492 boundary documents (27.2%) into secondary clusters.
* **Hybrid RRF (BM25F + Routed)**: Fuses the best lexical field-weighted model with soft cluster-routed neural search, achieving top benchmark effectiveness: **0.9104 Graded NDCG@10**, **1.0000 MRR@10**, and **0.5380 Recall@10**.

---

## Directory Layout

| Directory | Responsibility | Tracked Artifacts |
|---|---|---|
| `classical/` | Tokenization, BM25/BM25F scoring, multi-level inverted index | Source code, test suites |
| `neural/` | Sliding windows, Qwen3 embeddings, Manning Ch 6 zone weights, K-Means++ clustering & soft routing | Source code, models, search engine |
| `evaluation/` | 15 natural technical queries, pooled judgments, graded metrics, evaluation UI | Queries, consensus judgments, metrics CSVs, web UI |
| `data_inputs/` | Versioned query seeds, zone-weight judgments, classification seeds | Seed queries & judgments |
| `presentation/` | Official 14-slide academic benchmark presentation & publication charts | `IR_Benchmark_Presentation.pptx`, PNG charts |
| `filter/` | HTML extraction, content filtering, version deduplication, ZIP archive | Source code |
| `data/` | Common Crawl collection, validation, and repair | Source code |
| `scripts/` | Benchmark execution, graded evaluation, chart generation, deck updates | Utility scripts |

---

## Setup & Environment

Requires Python 3.11+:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the End-to-End Pipeline

Each component can be run independently or verified sequentially:

```bash
# 1. Classical Indexing & Retrieval (BM25 & BM25F)
python classical/run_pipeline.py --phase all

# 2. Neural Pipeline (Extraction, Windows, Embeddings, Clustering, Routing)
python neural/run_pipeline.py --step all

# 3. Full Benchmark Retrieval, Pooling & Graded Evaluation
python scripts/run_full_benchmark_and_graded_eval.py

# 4. Generate Presentation Charts
python scripts/generate_presentation_charts.py

# 5. Update and Format Presentation Deck
python scripts/update_presentation_deck.py
```

---

## Interactive Web UIs

### 1. Cross-Engine Search Comparison UI
Launches a side-by-side comparison interface (BM25 vs. BM25F vs. Dense vs. Neural Zones vs. Cluster-Routed):
```bash
python run_search_ui.py
# Open http://localhost:8080 in your browser
```

### 2. Evaluation & Judgment Adjudication UI
Launches the Cranfield evaluation interface for inspecting queries, pooled document rankings, and LLM consensus scores:
```bash
python evaluation/run_evaluation.py
# Open http://localhost:8000 in your browser
```

### 3. Neural-Only Standalone Search UI
```bash
python neural/run_pipeline.py --step 09
# Open http://localhost:8000 in your browser
```

---

## Presentation Deck

The canonical 14-slide academic presentation is maintained at:
* **Presentation PPTX**: [`presentation/IR_Benchmark_Presentation.pptx`](presentation/IR_Benchmark_Presentation.pptx)
* **High-Resolution Figures**: [`presentation/charts/`](presentation/charts/)
  * `cluster_category_alignment.png`: K-Means++ (k=8) domain purity & soft multi-membership expansion
  * `latency_throughput_pareto.png`: Query latency vs. Graded NDCG@10
  * `overall_effectiveness_hybrid.png`: Comprehensive 7-metric multi-system comparison
  * `precision_depth_curve.png`: Precision retention across cutoffs (P@1 to P@10)
  * `index_size_breakdown.png`: Corpus reduction & index storage footprint
