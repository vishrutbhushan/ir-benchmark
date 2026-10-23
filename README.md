# Technical IR Benchmark

An Information Retrieval benchmark comparing a hand-coded **Classic IR engine** (BM25 → zone-weighted BM25F) against a **Neural IR engine** (pretrained dense embeddings with exact matrix ranking) on a technical web documentation corpus.

---

## Setup

Activate your virtual environment and install dependencies:

```bash
# Using the existing virtual environment
.venv\Scripts\activate

# Install required packages
pip install -r requirements-technical-web-corpus.txt
pip install trafilatura langdetect matplotlib pillow
```

---

## Pipeline Execution

All pipeline phases are executed through the unified master runner `run_pipeline.py`. Each run automatically generates visual charts (`.png`), markdown summaries, and an interactive HTML dashboard (`outputs/dashboard.html`).

### 1. Phase 0 — Corpus Audit
Audits the raw HTML corpus for empty/JS-rendered bodies, language detection, error pages, and near-duplicate SimHash clusters:

```bash
# Run balanced audit across all 11 technical categories (equal representation)
python run_pipeline.py --phase 0 --sample-per-category 60 --sample-size 660
```
- **Outputs**: `outputs/phase0/audit_report.md`, `outputs/phase0/audit_report.json`, `outputs/phase0/charts/` (6 visual plots), `outputs/phase0/sample_inspections.jsonl`
- **Logs**: `logs/phase0/audit.log`

### 2. Phase 1 — Structured Extraction
Parses HTML pages into structured records (titles, heading hierarchy, sections, boilerplate-free body, isolated code blocks, and ISO dates):

```bash
# Run stratified extraction across all technology categories
python run_pipeline.py --phase 1 --stratified --max-docs 500

# Run full corpus extraction
python run_pipeline.py --phase 1 --corpus-root technical_web_corpus/technical_web_corpus --manifest-path technical_web_corpus/technical_web_corpus/metadata/corpus_manifest.jsonl
```
- **Outputs**: `outputs/phase1/structured_records.jsonl`, `outputs/phase1/corpus_statistics.md`, `outputs/phase1/corpus_statistics.json`, `outputs/phase1/charts/`
- **Logs**: `logs/phase1/extraction.log`

### 3. Phase 2 — Tokenization & Normalization
Tokenizes all structured records into per-zone token lists using Porter stemming, version-string preservation, and technical identifier expansion:

```bash
python run_pipeline.py --phase 2
```
- **Outputs**: `outputs/phase2/tokenized_corpus.jsonl`, `outputs/phase2/tokenization_stats.json`, `outputs/phase2/tokenization_stats.md`, `outputs/phase2/charts/` (4 visual plots)
- **Logs**: `logs/phase2/tokenization.log`

### 4. Run All Phases & Interactive Dashboard
Execute the end-to-end pipeline in sequence:

```bash
python run_pipeline.py --phase all
```
- **Visual Dashboard**: Open `outputs/dashboard.html` in any browser for an interactive, graphical overview of corpus health, extraction, and tokenization metrics.

---

## Directory Structure

```text
ir-benchmark/
├── run_pipeline.py            # Master CLI runner
├── src/                       # Modular pipeline source code
│   ├── phase0_audit/          # Corpus sampling, validation, and near-duplicate detection
│   ├── phase1_extraction/     # HTML parsing, boilerplate removal, and statistics
│   ├── phase2_tokenization/   # Tokenizer, Porter stemmer, normalizer, and tests
│   └── utils/                 # Logging, file I/O, SimHash, and chart visualizer
├── outputs/                   # Hierarchical phase outputs, charts, and dashboard.html
│   ├── dashboard.html         # Interactive visual analytics dashboard
│   ├── phase0/charts/         # Phase 0 exclusion & length charts (.png)
│   ├── phase1/charts/         # Phase 1 category & code prevalence charts (.png)
│   └── phase2/charts/         # Phase 2 token distribution & vocabulary charts (.png)
├── logs/                      # Hierarchical execution logs
├── queries.jsonl              # 25 technical queries across 7 categories
└── technical_web_corpus/      # Raw HTML documents and crawl metadata
```
