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
pip install trafilatura langdetect
```

---

## Pipeline Execution

All pipeline phases are executed through the unified master runner `run_pipeline.py`.

### 1. Phase 0 — Corpus Audit
Audits the raw HTML corpus for empty/JS-rendered bodies, language detection, error pages, and near-duplicate SimHash clusters:

```bash
# Run audit on a sample of 500 documents across technologies
python run_pipeline.py --phase 0 --sample-size 500
```
- **Outputs**: `outputs/phase0/audit_report.md`, `outputs/phase0/audit_report.json`, `outputs/phase0/sample_inspections.jsonl`
- **Logs**: `logs/phase0/audit.log`

### 2. Phase 1 — Structured Extraction
Parses HTML pages into structured records (titles, heading hierarchy, sections, boilerplate-free body, isolated code blocks, and ISO dates):

```bash
# Run stratified extraction across all technology categories
python run_pipeline.py --phase 1 --stratified --max-docs 500

# Run full corpus extraction
python run_pipeline.py --phase 1
```
- **Outputs**: `outputs/phase1/structured_records.jsonl`, `outputs/phase1/corpus_statistics.md`, `outputs/phase1/corpus_statistics.json`
- **Logs**: `logs/phase1/extraction.log`

### 3. Run All Phases
Execute the end-to-end pipeline in sequence:

```bash
python run_pipeline.py --phase all
```

---

## Directory Structure

```text
ir-benchmark/
├── run_pipeline.py            # Master CLI runner
├── src/                       # Modular pipeline source code
│   ├── phase0_audit/          # Corpus sampling, validation, and near-duplicate detection
│   ├── phase1_extraction/     # HTML parsing, boilerplate removal, and statistics
│   └── utils/                 # Logging, file I/O, and SimHash utilities
├── outputs/                   # Hierarchical phase outputs (reports, JSONL records)
├── logs/                      # Hierarchical execution logs
└── technical_web_corpus/      # Raw HTML documents and crawl metadata
```
