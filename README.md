# IR Benchmark: Classical BM25 vs. Dense Neural vs. Hybrid RRF

A rigorous information retrieval benchmark evaluating classical lexical matching (BM25, BM25F), dense neural retrieval (Qwen3-Embedding-0.6B), unsupervised cluster routing (K-Means++ with soft multi-membership), and Reciprocal Rank Fusion (RRF) across a 45,881-document technical engineering corpus.

---

## Overview & Retrieval Paradigms

The benchmark evaluates five distinct retrieval paradigms on complex technical documentation, code snippets, and conversational engineering queries:

1. **Flat BM25 (Lucene Default)**: Baseline term-matching with standard BM25 ($k_1=1.2, b=0.75$) over raw concatenated document text.
2. **Hierarchical BM25F (Fields)**: Zone-weighted BM25 incorporating document structure (Title weight: 3.0, Headings: 2.0, Code blocks: 2.0, Body: 1.0).
3. **Qwen3 Dense (Exhaustive)**: Dense semantic retrieval using Qwen3-Embedding-0.6B (512-token sliding windows, 64-token overlap, cosine similarity).
4. **Cluster-Routed Neural (Soft Boundary)**: Unsupervised K-Means++ clustering ($k=8$) over document representations with soft multi-membership ($\Delta \le 0.03$), pruning vector scan by 55.5% while indexing 12,492 boundary documents (27.2%) into secondary clusters.
5. **Hybrid RRF (BM25F + Routed)**: Reciprocal Rank Fusion combining top lexical field matching with fast cluster-routed neural search ($k_{\text{rrf}}=60$).

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

## Running the Pipeline

Each pipeline stage can be executed independently or run end-to-end:

```bash
# 1. Full Pipeline Execution
python run_full_pipeline.py

# 2. Classical Pipeline Only (Tokenization, BM25, and BM25F)
python classical/run_pipeline.py --phase all

# 3. Neural Pipeline Only (Extraction, Windows, Embeddings, Clustering, Routing)
python neural/run_pipeline.py --step all
```

---

## Interactive Web UIs

### 1. Cross-Engine Search Comparison UI
Launches a side-by-side search comparison interface evaluating BM25, BM25F, Dense, Neural Zones, and Cluster-Routed retrieval:
```bash
python run_search_ui.py
# Open http://localhost:8080 in your browser
```

### 2. Evaluation & Judgment Adjudication UI
Launches the Cranfield evaluation interface for inspecting queries, pooled rankings, and consensus scores:
```bash
python evaluation/run_evaluation.py
# Open http://localhost:8000 in your browser
```

---

## Evaluation Methodology

The benchmark follows the classical **Cranfield evaluation paradigm**:
- **Query Set**: 15 technical engineering queries spanning exact syntax commands, error trace diagnosis, and conversational architecture queries.
- **Pooling & Adjudication**: Top-10 results from each retrieval engine were pooled (384 unique query-document pairs) and independently evaluated by two LLM judges:
  - **Judge A**: NVIDIA Nemotron-3.5-Bolt
  - **Judge B**: Google Gemma 4:12b
  - **Observed Agreement**: 71.9% (276 / 384 concordant pairs), with Cohen's $\kappa = 0.4740$ (Moderate Agreement).
- **Consensus Relevance Scale**:
  - **Grade 2 (Highly Relevant)**: Concordant relevance (both judges agree: 119 pairs).
  - **Grade 1 (Partially Relevant)**: Discordant relevance (one judge marks relevant: 108 pairs).
  - **Grade 0 (Not Relevant)**: Concordant non-relevance (both judges agree: 157 pairs).
- **Graded Discounting**: Graded NDCG uses standard $2^g - 1$ gain discounting to reward top-tier primary documentation over peripheral matches.

---

## Benchmark Results

### Comprehensive 7-Metric Benchmark Evaluation Across All 15 Complex & Conversational Queries

| Method / Paradigm | Graded NDCG@5 | Graded NDCG@10 | P@5 | P@10 | Recall@10 | MRR@10 | Binary NDCG@10 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flat BM25 (Lucene Default)** | 0.7212 | 0.8600 | 0.7867 | 0.7000 | 0.5176 | 0.9333 | 0.9255 | 31.0 ms |
| **Hierarchical BM25F (Fields)** | 0.7406 | 0.8748 | 0.7333 | 0.6800 | 0.5010 | 0.9556 | 0.9237 | 43.5 ms |
| **Qwen3 Dense (Exhaustive)** | 0.7252 | 0.8639 | 0.8133 | 0.7333 | 0.5214 | 0.9222 | 0.9253 | 53.7 ms |
| **Cluster-Routed Neural (Soft)** | **0.8042** | 0.9088 | **0.8667** | 0.6867 | 0.4987 | 0.9667 | 0.9446 | **23.3 ms** |
| **Hybrid RRF (BM25F + Routed)** | 0.7690 | **0.9104** | 0.8133 | **0.7333** | **0.5380** | **1.0000** | **0.9547** | 66.8 ms |

### Key Findings
* **Cluster Routing Delivers Top Latency & Quality**: Cluster-routed neural search achieves **23.3 ms** latency (2.3× speedup over exhaustive dense scan) and **0.9088 Graded NDCG@10** while scanning only 44.5% of the vector space.
* **Hybrid RRF Achieves Peak Effectiveness**: Fusing BM25F and cluster-routed neural retrieval achieves the highest overall benchmark score: **0.9104 Graded NDCG@10**, **1.0000 MRR@10**, and **0.5380 Recall@10**.
* **Lexical vs. Neural Complementarity**: Exact syntax queries (flags, configuration options) heavily favor BM25F, while conversational intent queries heavily favor dense representations. Hybrid RRF combines the strengths of both without parameter tuning.

---

## Presentation Deck & Publication Figures

The canonical 14-slide academic presentation is maintained at:
* **Presentation PPTX**: [`presentation/IR_Benchmark_Presentation.pptx`](presentation/IR_Benchmark_Presentation.pptx)
* **High-Resolution Figures**: [`presentation/charts/`](presentation/charts/)
  * `cluster_category_alignment.png`: K-Means++ (k=8) domain purity & soft multi-membership expansion
  * `latency_throughput_pareto.png`: Query latency vs. Graded NDCG@10
  * `overall_effectiveness_hybrid.png`: Comprehensive 7-metric multi-system comparison
  * `precision_depth_curve.png`: Precision retention across cutoffs (P@1 to P@10)
  * `index_size_breakdown.png`: Corpus reduction & index storage footprint

---

## Directory Layout

| Directory | Responsibility | Tracked Artifacts |
|---|---|---|
| `classical/` | Tokenization, BM25/BM25F scoring, multi-level inverted index | Source code, test suites |
| `neural/` | Sliding windows, Qwen3 embeddings, Manning zone weights, K-Means++ clustering & soft routing | Source code, models, search engine |
| `evaluation/` | 15 technical queries, pooled judgments, graded metrics, evaluation UI | Queries, consensus judgments, metrics CSVs, web UI |
| `presentation/` | Official 14-slide academic benchmark presentation & publication charts | `IR_Benchmark_Presentation.pptx`, PNG charts |
| `filter/` | HTML extraction, content filtering, version deduplication, ZIP archive | Source code |
| `data/` | Common Crawl collection, validation, and repair | Source code |
