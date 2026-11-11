# Classical IR & BM25 Engine Architecture Documentation

This document provides a comprehensive technical overview of the hand-coded **Classical Information Retrieval (IR) Engine** built for the **61,966-document** deduplicated technical web corpus benchmark.

---

## 1. System Philosophy & Constraints

* **Zero-Dependency Policy:** The core indexing structures, posting lists, scoring algorithms, and evaluation metrics are hand-coded from scratch in Python with zero external search libraries (**No Lucene, Whoosh, Rank-BM25, or Elasticsearch**).
* **Multi-Zone Awareness:** Technical documents are parsed into four semantic zones (`title`, `heading`, `code`, `body`), each carrying distinct relevance weights.
* **Exact & SimHash Near-Duplicate Removal (`src/utils/simhash.py`):**
  - Raw manifest pages: `77,817`
  - Short / boilerplate stubs removed (`< 200` chars): `252` (`77,565` extracted)
  - Exact SHA-256 content duplicates removed: `4,014`
  - Cross-version 64-bit SimHash near-duplicates removed (Hamming distance $\le 3$): `11,585`
  - **Final Unique Indexed Documents:** **`61,966`**
* **Corpus Scale:**
  - **Indexed Documents:** `61,966`
  - **Vocabulary Size:** `1,799,233` unique stemmed terms
  - **Total Posting Entries:** `14,338,715`
  - **Total Positional Entries:** `72,795,150`
  - **Passage Sub-Index:** `652,089` structural sections

---

## 2. End-to-End Pipeline Data Flow

```text
                               Raw HTML Corpus
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   Phase 1: HTML Extraction    │
                      │       [html_parser.py]        │
                      └───────────────┬───────────────┘
                                      │  (title, headings, code, body)
                                      ▼
                      ┌───────────────────────────────┐
                      │    Phase 2: Tokenization      │
                      │        [tokenizer.py]         │
                      └───────────────┬───────────────┘
                                      │  (Porter stemmed, atomic versions)
                                      ▼
  Query String        ┌───────────────────────────────┐
       │              │  Phase 3: Inverted Index Core │
       ▼              │      [inverted_index.py]      │
┌──────────────┐      ├───────────────────────────────┤
│    Query     │      │ • Level 0: Taxonomy Directory │
│  Processor   │─────▶│ • Level 1: Multi-Zone Index   │
└──────────────┘      │ • Level 2: Section Sub-Index  │
                      │ • Tier-1 Champion Lists       │
                      └───────────────┬───────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     ┌────────────────────┐                        ┌────────────────────┐
     │   Flat BM25        │                        │   Zone BM25F       │
     │  [bm25_scorer.py]  │                        │ [bm25f_scorer.py]  │
     └─────────┬──────────┘                        └─────────┬──────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │   Multi-Level Search    │
                         │  [multilevel_index.py]  │
                         └────────────┬────────────┘
                                      │  (Doc + Best Section Snippet)
                                      ▼
                         ┌─────────────────────────┐
                         │   Benchmark Runner      │
                         │ [classic_ir_runner.py]  │
                         └─────────────────────────┘
```

---

## 3. Detailed Component Breakdown

### 3.1. Phase 1: Structured HTML Extraction (`src/phase1_extraction/`)
Technical HTML pages are stripped of web chrome, navbars, and cookie banners using `trafilatura` and decomposed into discrete zones:
- `title`: Extracted from `<title>` and main `<h1>`.
- `headings`: Structural table of contents (`<h2>`–`<h6>`).
- `code_text`: Command lines, config blocks, and code extracted from `<pre>` and `<code>` tags.
- `body_text`: Explanatory prose and paragraphs.
- `dates`: ISO-normalized dates separated from free text.

### 3.2. Phase 2: Technical Linguistic Normalization (`src/phase2_tokenization/`)
Standard NLP tokenizers fail on code tokens. The technical tokenizer applies:
- **Porter Stemmer:** Uniform algorithmic stemming across both documents and search queries.
- **Atomic Version Protection:** Preserves strings like `3.14.7` and `v2.4` without breaking on dots.
- **Compound Identifier Dual Indexing:** Hyphenated and camelCase tokens (`react-router-dom`) are indexed both as atomic terms and as split sub-tokens (`react`, `router`, `dom`).

### 3.3. Phase 3: The Multi-Zone Inverted Index (`src/phase3_classic_ir/inverted_index.py`)
Rather than storing a single scalar frequency, every posting stores a zone frequency vector:

```python
@dataclass
class Posting:
    doc_id: str
    zone_tfs: dict[str, int]         # {'title': 1, 'heading': 2, 'code': 3, 'body': 5}
    positions: dict[str, list[int]]  # Word offsets per zone
```

#### Multi-Tier Hierarchy:
- **Level 0 (Taxonomy Directory):** Inverted mappings for 13 Categories and 76 Technologies allowing fast category-constrained sub-index retrieval.
- **Level 1 (Document Zonal Index):** 61,966 deduplicated full documents with 4 zones.
- **Level 2 (Section Passage Index):** 652,089 individual document sections for passage retrieval.
- **Tier-1 Champion Lists:** Top-50 documents per term pre-filtered by prominent zone presence and frequency to accelerate candidate generation.

---

## 4. Mathematical Ranking Formulas

### 4.1. Flat Okapi BM25 (`bm25_scorer.py`)
Treats the entire document as a single flat field by summing zone frequencies:

$$\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

- **Robertson-Spärck Jones IDF:**
  $$\text{IDF}(t) = \ln\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1.0\right)$$
- **Default Hyperparameters:** $k_1 = 1.2$, $b = 0.75$.

---

### 4.2. Zone-Weighted BM25F (`bm25f_scorer.py`)
Computes normalized pseudo-frequencies $\tilde{f}(t, d)$ across independent zones before applying saturation:

$$\tilde{f}(t, d) = \sum_{z \in \{\text{title}, \text{heading}, \text{code}, \text{body}\}} w_z \cdot \frac{\text{tf}_z(t, d)}{(1 - b_z) + b_z \cdot \left(\frac{|d_z|}{\text{avgdl}_z}\right)}$$

$$\text{Score}_{\text{BM25F}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\tilde{f}(t, d)}{k_1 + \tilde{f}(t, d)}$$

#### Zone Configuration:
| Zone ($z$) | Weight ($w_z$) | Normalization ($b_z$) | Corpus Average Length ($\text{avgdl}_z$) |
| :--- | :---: | :---: | :--- |
| **`title`** | **$4.0$** | $0.30$ | ~7.6 tokens |
| **`heading`** | **$2.5$** | $0.50$ | ~38.4 tokens |
| **`code`** | **$2.0$** | $0.60$ | ~289.0 tokens |
| **`body`** | **$1.0$** | $0.75$ | ~896.6 tokens |

#### Hierarchical Taxonomy Conditioning:
Documents whose metadata matches the technology or category inferred from the query receive a principled boost:
$$\text{Score}_{\text{final}}(q, d) = \text{Score}_{\text{BM25F}}(q, d) \times (1 + \text{hierarchical\_boost})$$
*(Default boost: $+25\%$ for direct technology match, $+12.5\%$ for category match).*

---

## 5. Query Processing & Typo Resilience (`query_processor.py`)

Incoming queries are normalized through a 3-step pipeline:
1. **Boolean Filtering:** Strips explicit `NOT <clause>` strings so negated terms are excluded from positive accumulator scoring.
2. **Technical Tokenization:** Applies identical Porter stemming and version protection.
3. **Levenshtein Spell Correction:** For out-of-vocabulary or low-frequency tokens, queries are matched against vocabulary frequency buckets using Levenshtein distance ($\le 2$):
   - Example: `pytorh` $\to$ `pytorch`
   - Example: `dockr` $\to$ `docker`
   - Example: `kubernets` $\to$ `kubernetes`

---

## 6. Multi-Level Document & Passage Search (`multilevel_index.py`)

1. **Document Retrieval:** BM25F retrieves the top-$K$ candidate documents.
2. **Passage Scoring:** Within the top candidate documents, Level 2 sections are evaluated against query terms.
3. **Result Presentation:** Returns the document URL and title alongside the highest-scoring passage heading, level, and preview snippet.

---

## 7. File Structure Reference

```text
src/phase3_classic_ir/
├── inverted_index.py     # Posting dataclass, DF/IDF calculation, Champion lists, index persistence
├── bm25_scorer.py        # Flat Okapi BM25 implementation (k1=1.2, b=0.75)
├── bm25f_scorer.py       # Multi-zone BM25F with length normalization and hierarchical conditioning
├── multilevel_index.py   # Two-tier Document + Section passage retrieval
├── query_processor.py    # Query stemming, Boolean NOT cleanup, and Levenshtein typo correction
├── classic_ir_runner.py  # Benchmark execution, comparative report generation, and charts
└── test_classic_ir.py    # Unit tests with hand-calculated precision checks
```

