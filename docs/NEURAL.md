# Neural retrieval

## Scope and status

The teacher's assignment compares classic keyword ranking with open-source neural
embeddings ranked by cosine, using 20 challenge queries, NDCG@5, MAP and failure
analysis. This branch covers the neural engine and shared corpus repairs. Classic
ranking belongs to the other engine.

The implementation includes the four-zone build, weight learning, both classifiers,
routing, UI and evaluation code. **The full build, fitting and engineering
verification have completed.** Engineering checks do not establish retrieval quality. Final qrels and the
classic-versus-neural failure report remain unfinished; no improvement is claimed.

## Data and preprocessing

The original ZIP contains 79,268 captures. The corrected local corpus contains
68,414 documents across 76 technologies and 11 original categories. Repairs replace
HTML at its normal path; exclusions leave the working manifest and remain locally
recoverable. The original archive is preserved. Both engines must use this same
corrected snapshot, not independently recovered versions.

Records contain `doc_id`, URL, title, headings, sections, body, code, date mentions
and metadata. Generic content selectors remove navigation and boilerplate. The
extractor preserves `<pre>` and standalone multiline `<code>` indentation. Inline
code stays in prose. Tables become plain visible text with spaces between cells;
there is no table reconstruction, language-specific code parser or code execution.
A retained RFC rule distinguishes preformatted prose from code.

Neural text keeps case, stopwords, negation, punctuation, versions and identifiers.
It uses the model tokenizer, without stemming or lemmatization. Dates are mentions,
not verified publication dates. Code is embedded as text in its own zone with the
same model. Exact identifiers, negation and version constraints can still fail.

## Index and query pipeline

```text
Corrected HTML and manifest
  → structured extraction in SQLite
  → independent title / headings / body / code windows
  → pinned encoder → normalized vectors in four NumPy stores
  → refresh learning tables → fit weights → train classifiers → tune routing

Query → retrieval instruction → query embedding
  → optional class routing and explicit user filters
  → hand-coded cosine → per-document zone maxima
  → dense or weighted ranking → snippets → pagination
```

Each zone uses up to 496 content tokens, 64-token overlap and a checked maximum
of 512 total input tokens. Windows contain source text only; no synthetic title or
technology prefix. Long zones are fully windowed. The current preparation produces
68,414 title, 61,511 heading, 255,445 body and 91,536 code windows: 476,906 total.

The encoder is `Qwen/Qwen3-Embedding-0.6B`, revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, producing 1,024 dimensions. Queries use
its retrieval instruction; documents do not. Pooling selects the last real token
and L2-normalizes. The pinned causal decoder permits right padding without a
redundant attention mask for the pooled real-token positions.

FP16 vectors persist on disk (about 0.91 GiB for this prepared index). Scoring uses
FP32, renormalizes stored vectors and computes matrix-vector cosine directly.
SQLite stores text/metadata; NumPy memory maps support block scans, with GPU
residence when memory permits. No vector database, ANN library or external service
is needed. Small bounded caches reuse query embeddings and rankings; pagination
loads text only for visible results.

## Four retrieval modes

| Mode | Document ranking |
|---|---|
| `dense` | Maximum window cosine across all zones; comparison baseline |
| `zones` | Learned weighted sum of per-zone maximum cosines |
| `routed-centroid` | Predict source classes with centroids, then weighted-zone retrieval |
| `routed-nb` | Predict source classes with multinomial Naive Bayes, then weighted-zone retrieval |

The UI defaults to `routed-centroid`; report the plain dense baseline separately.
Explicit technology/category filters restrict routing candidates. Ties use document
IDs, and equal window scores choose the earliest window. Fixed vectors give
reproducible ranking; GPU rebuilding is not promised bitwise identical.

Weights are nonnegative and sum to one. A deterministic grid with step 0.05
minimizes training-only squared error against binary relevance. Scores renormalize
over present zones; if those weights sum to zero, they use equal present-zone
weights. Training and retrieval use the same function. Max pooling favors long
documents by giving them more opportunities to match.

The current fit assigns title/headings/body/code weights of **0/0/1/0**. This does
not demonstrate a benefit from combining zones; final retrieval evaluation is pending.

## Learning tables and category routing

| Table | Contents and use |
|---|---|
| `data/neural/zone_weights/training.csv` | Query/document details, binary relevance, split, four cosines and presence flags; 262 pairs, 44 queries split 33 train / 11 validation |
| `data/neural/classification/training.csv` | Full document text, title, class, split and duplicate-group ID; 420 documents, 20/5/5 per class |
| `classes.csv`, `technology_classes.csv` in the classification folder | Fixed class definitions and complete technology-to-class mapping |

The 14 domains are cloud platforms, AI/ML compute, MLOps orchestration, programming
languages, databases/storage, search engines, messaging/RPC, distributed data,
application frameworks, containers/orchestration, DevOps/build/release,
OS/networking/web, security/identity and observability.

Class labels describe source domains, not independent semantic topics or relevance.
Canonical version URL families and identical extracted content stay within one
classification split. The build refreshes classification text and zone features
from the production index. Only training examples fit model parameters.

Naive Bayes uses training token counts, training-only vocabulary (df ≥ 2), add-one
smoothing and empirical priors. Centroids average normalized training-document
vectors; each document vector averages its pure-zone window embeddings. Both share
the same labeled examples. Validation/test accuracy, macro-F1 and confusion matrices
are calculated directly. NB posterior values are not calibrated relevance scores.

On the 70-document held-out test split, centroid accuracy is 84.3% (macro-F1 0.832)
and NB accuracy is 74.3% (macro-F1 0.727). These measure source-label agreement.

Routing selects the smallest number of top classes retaining mean 98% of exhaustive
weighted top-20 results, with at least 90% for each development validation query.
This measures reference-ranking recall, not relevance recall. All 14 classes may
be required; speedup is measured rather than assumed. Source-class partitions are
predefined: this is supervised classification, not unsupervised clustering or RAPTOR.
The summary hierarchy is completely removed.

## Queries, evaluation and checks

The repository originally had 25 candidate queries. The current 20-query suite was
revised for technical ambiguity and retained-source support. It includes three
negation, four homonym, four temporal, four exact-terminology, four multiple-constraint
and one paraphrase query. Each has a judgment intent. The 44 `DEV...` queries were
created separately for learning and are not final test queries.

After freezing the build, `evaluation-pool` unions dense/weighted top-10 and routed
top-5 candidates with source excerpts. Grades are 0 (irrelevant), 1 (substantive
partial answer), 2 (direct answer satisfying the constraints). They must come from
source evidence, not candidate rank. Final `data/evaluation/qrels.csv` is pending.

Evaluation uses full returned rankings, exponential-gain NDCG@5 and binary AP
(grade > 0), averaged across queries for MAP. Unjudged documents receive zero gain;
pooled judgments limit MAP interpretation. Classic results must use the same query
hash, corpus IDs and qrels, extending the pool with their top-20 before comparison.
Report ranking depth, judged coverage and failures by query type.

Verification checks every window's source offsets, vector/document mappings,
normalization, GPU-versus-NumPy cosine agreement, all query/mode combinations,
filters and cached pagination. Raw corpus validation checks manifest hashes and
coverage. Checkpoints flush completed vectors before advancing progress. Interrupted
builds repeat only the unfinished checkpoint block; completed unchanged zones reuse
their saved files. Keep local outputs to resume.

## Course connections and source map

- Chapter 6: cosine/vector scoring and learned weighted zones. Our continuous
  cosine features adapt the book's Boolean zone-match table; they are not its exact
  binary example.
- Chapter 8: NDCG/MAP and common relevance judgments.
- Chapters 13–14 and the text-mining slides: fixed classes, labeled documents,
  Naive Bayes priors/likelihoods/smoothing and Rocchio-style centroid classification.
- Classification routing reduces candidate space; no claim of clustering or
  summary-tree indexing is made.

`run_pipeline.py` and `src/neural/pipeline.py` orchestrate stages.
`src/corpus_validation/` repairs the shared corpus; `src/neural/preprocessing/`
extracts and windows it. `zone_index.py`, `model.py`, `embedding.py` build stores.
`learning/`, `routing.py`, `scoring.py`, `search.py` implement learning/retrieval.
`evaluation/`, `verification.py`, `audit.py` provide checks and metrics.
`server.py` and `web/` provide the UI. See README for commands.

References: [weighted-zone learning](https://nlp.stanford.edu/IR-book/html/htmledition/learning-weights-1.html),
[Naive Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html),
[Rocchio classification](https://nlp.stanford.edu/IR-book/html/htmledition/rocchio-classification-1.html),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
