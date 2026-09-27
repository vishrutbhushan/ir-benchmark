# Phase 0 — Corpus Audit Report

**Sample Size Analyzed:** 1090 documents across technologies.
**Text-Length Threshold:** < 200 characters.

## Visual Charts & Graphs

![Phase 0 Exclusions Breakdown](charts/phase0_exclusion_breakdown.png)

![Phase 0 Document Lengths](charts/phase0_tech_lengths.png)

## 1. Exclusion Categories Summary

| Category | Flagged Count | Sample % | Example Doc IDs |
| :--- | :--- | :--- | :--- |
| **Empty / Short Body (<200 chars)** | 14 | 1.28% | document_068019, document_073277, document_075145, document_057029, document_067935 |
| **Client-Side JS Render Stubs** | 0 | 0.0% | None |
| **Non-HTML Content Mislabeled** | 3 | 0.28% | document_073277, document_075145, document_074414 |
| **HTTP 200 Error / 404 Pages** | 0 | 0.0% | None |
| **Non-English Documents** | 60 | 5.5% | document_000102, document_043203, document_009708, document_054531, document_074158 |
| **Clean / Valid Documents** | 1016 | 93.21% | — |

## 2. Near-Duplicate Analysis (SimHash Clustering)

- **Documents Analyzed for Near-Duplicates:** 1076
- **Near-Duplicate Clusters Found:** 18
- **Total Documents in Duplicate Clusters:** 51

### Top Near-Duplicate Cluster Samples

* **Cluster 1** (2 docs) — Representative: `document_018971` (`Web | gRPC`)
  - Members: document_018971, document_018961
* **Cluster 2** (2 docs) — Representative: `document_029194` (`Unusual Network Connection to Suspicious Web Service | Prebuilt detection rules reference`)
  - Members: document_029194, document_029626
* **Cluster 3** (3 docs) — Representative: `document_002454` (`venv â€” Creation of virtual environments â€” Python 3.14.7 documentation`)
  - Members: document_002454, document_002594, document_002739
* **Cluster 4** (3 docs) — Representative: `document_075473` (`round â€” Ray 2.58.0`)
  - Members: document_075473, document_076089, document_076313
* **Cluster 5** (6 docs) — Representative: `document_037118` (`Branching and Tagging - Apache Icebergâ„¢`)
  - Members: document_037118, document_037016, document_037001, document_037122, document_036980, document_037050

## 3. Technology Distribution & Character Lengths (Top 15 Technologies in Sample)

| Technology | Sample Docs | Avg Char Length | Short (<200) | Non-English | Errors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Java | 15 | 3030.9 | 0 | 9 | 0 |
| Spring Boot | 15 | 1924.1 | 0 | 0 | 0 |
| RFCs | 15 | 140880.7 | 7 | 0 | 0 |
| Rust | 15 | 5830.8 | 1 | 0 | 0 |
| NIST | 15 | 31277.4 | 0 | 0 | 0 |
| Kubeflow | 15 | 10123.5 | 0 | 0 | 0 |
| React | 15 | 11158.0 | 0 | 2 | 0 |
| Elasticsearch | 15 | 57060.5 | 0 | 0 | 0 |
| ClickHouse | 15 | 9018.1 | 0 | 0 | 0 |
| OpenSSL | 15 | 81377.7 | 0 | 0 | 0 |
| Linkerd | 15 | 8423.7 | 0 | 0 | 0 |
| Jaeger | 15 | 4743.5 | 0 | 0 | 0 |
| Google Cloud | 15 | 23352.4 | 0 | 8 | 0 |
| Python | 15 | 36411.4 | 0 | 0 | 0 |
| Oracle Cloud | 15 | 6253.2 | 0 | 0 | 0 |

## 4. Key Recommendations for Human Sign-off

1. **Empty/JS Stub Exclusion:** Exclude all documents with body text < 200 characters or explicit JS stub indicators (`<div id='root'></div>` with no extracted text).
2. **Error / 404 Filter:** Drop documents matching error/redirect page signatures.
3. **Language Filter:** Require English text (`en` detected, or manifest `eng` when text is too short).
4. **Duplicate Deduplication:** Retain only the first/representative document from each near-duplicate SimHash cluster.
5. **Spot-Check File:** Review `outputs/phase0/sample_inspections.jsonl` to verify flagged samples.
