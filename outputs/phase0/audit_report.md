# Phase 0 — Corpus Audit Report

**Sample Size Analyzed:** 500 documents across technologies.
**Text-Length Threshold:** < 200 characters.

## 1. Exclusion Categories Summary

| Category | Flagged Count | Sample % | Example Doc IDs |
| :--- | :--- | :--- | :--- |
| **Empty / Short Body (<200 chars)** | 4 | 0.8% | document_068019, document_073277, document_075145, document_057029 |
| **Client-Side JS Render Stubs** | 0 | 0.0% | None |
| **Non-HTML Content Mislabeled** | 2 | 0.4% | document_073277, document_075145 |
| **HTTP 200 Error / 404 Pages** | 0 | 0.0% | None |
| **Non-English Documents** | 26 | 5.2% | document_000102, document_043203, document_009708, document_054531, document_074158 |
| **Clean / Valid Documents** | 470 | 94.0% | — |

## 2. Near-Duplicate Analysis (SimHash Clustering)

- **Documents Analyzed for Near-Duplicates:** 496
- **Near-Duplicate Clusters Found:** 4
- **Total Documents in Duplicate Clusters:** 9

### Top Near-Duplicate Cluster Samples

* **Cluster 1** (2 docs) — Representative: `document_002454` (`venv — Creation of virtual environments — Python 3.14.7 documentation`)
  - Members: document_002454, document_002594
* **Cluster 2** (2 docs) — Representative: `document_037118` (`Branching and Tagging - Apache Iceberg™`)
  - Members: document_037118, document_037016
* **Cluster 3** (2 docs) — Representative: `document_078950` (`nvmath.bindings.cublas.swap_ex_64 — nvmath-python`)
  - Members: document_078950, document_077440
* **Cluster 4** (3 docs) — Representative: `document_047622` (`Rules | Cloudflare API`)
  - Members: document_047622, document_045763, document_046607

## 3. Technology Distribution & Character Lengths (Top 15 Technologies in Sample)

| Technology | Sample Docs | Avg Char Length | Short (<200) | Non-English | Errors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pulsar | 12 | 6862.2 | 0 | 0 | 0 |
| Django | 11 | 8677.3 | 0 | 0 | 0 |
| Helm | 11 | 5442.5 | 0 | 0 | 0 |
| Java | 10 | 3490.4 | 0 | 5 | 0 |
| Kubeflow | 10 | 11096.7 | 0 | 0 | 0 |
| Linux Kernel | 10 | 5405.9 | 0 | 0 | 0 |
| ClickHouse | 9 | 9697.7 | 0 | 0 | 0 |
| FastAPI | 9 | 34450.6 | 0 | 4 | 0 |
| OpenSearch | 9 | 19266.8 | 0 | 0 | 0 |
| MongoDB | 9 | 13355.4 | 0 | 0 | 0 |
| Podman | 9 | 5688.4 | 0 | 0 | 0 |
| Node.js | 9 | 82435.1 | 0 | 0 | 0 |
| NIST | 8 | 38506.2 | 0 | 0 | 0 |
| Linkerd | 8 | 7631.2 | 0 | 0 | 0 |
| Python | 8 | 24940.9 | 0 | 0 | 0 |

## 4. Key Recommendations for Human Sign-off

1. **Empty/JS Stub Exclusion:** Exclude all documents with body text < 200 characters or explicit JS stub indicators (`<div id='root'></div>` with no extracted text).
2. **Error / 404 Filter:** Drop documents matching error/redirect page signatures.
3. **Language Filter:** Require English text (`en` detected, or manifest `eng` when text is too short).
4. **Duplicate Deduplication:** Retain only the first/representative document from each near-duplicate SimHash cluster.
5. **Spot-Check File:** Review `outputs/phase0/sample_inspections.jsonl` to verify flagged samples.
