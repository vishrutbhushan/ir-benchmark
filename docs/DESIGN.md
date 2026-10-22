# Neural retrieval design

Current search: HTML → structured sections → overlapping token passages → pinned
Qwen3-Embedding-0.6B → hand-coded cosine → maximum passage per document → pagination.
SQLite stores text/metadata; NumPy stores vectors on disk. The summary tree is removed.

## Shared dataset

Use `technical_web_corpus/metadata/corpus_manifest.jsonl` and its HTML files directly.
Accepted repairs replace bad captures at their ordinary paths; excluded HTML is moved
out of the working dataset. No companion allowlist is needed. The original ZIP and
`outputs/corpus_backup/` provide recovery. `--phase validate` checks every corrected
HTML hash, size and exact manifest/file coverage. This verifies bytes and membership,
not perfect semantic extraction quality. Neural and classic extraction remain separate.

## Learning tables and the book

| Task | Book-style supervised example | Our CSV |
|---|---|---|
| Zone weights, Chapter 6 | query, document, zone scores, relevance | `data/neural/zone_weights/training.csv` |
| NB / centroid, Chapters 13–14 | document text, class label | `data/neural/classification/training.csv` |

Each CSV is self-contained. IDs and split columns support reproducibility. Class
names are included beside class IDs; query text is repeated beside query IDs.
Definitions and the technology/class mapping are reference tables, not required
joins for training. Evidence and earlier normalized tables remain in ignored history.

The book's Figure 6.5 lists query/document pairs, zone match scores and relevance.
Our zone table follows that row structure. Its introductory example uses Boolean term matches. Our scores are
cosines, so this is a neural adaptation of weighted zone scoring, not an exact
reproduction of its binary example. For each zone, take the maximum query/window
cosine. Learn four nonnegative weights summing to one; renormalize over present zones.
Do not use class labels, URLs or source metadata as relevance features.

Four fixed development zones: title, headings, body and code. Visible table text
joins body; code whitespace is preserved. These features cover all windows of the
selected documents. Production search still uses contextual section passages:
separate production zone matrices and weight fitting have not been implemented.

Zone training has 44 queries / 264 pairs, with 262 usable pairs. All pairs belonging
to a query stay in one split: 33 training queries, 11 validation queries. Unjudged
documents are not negatives. Keep final challenge queries out of weight fitting.

Classification has 14 source-domain classes and 420 labeled documents: 20 train,
5 validation, 5 test per class. Both methods use exactly the same rows. Labels follow
the existing technology/domain mapping; documents may discuss several topics.
NB learns training-only vocabulary, priors and smoothed term counts. Centroids use
clean training-document embeddings. Use only `text` as model input; never class/split
columns or retrieval's added Technology prefixes. Check macro-F1/per-class results.
Small labeled subsets are enough to begin; held-out results decide whether to expand.
Classifiers and routing have not been implemented. Source-domain classification
accuracy alone would not establish routing quality for short user queries.

## Boundaries

No stemming, stopword removal, code execution, AST parsing, OCR, generated answers,
summary tree, or vector database in the neural path. Preserve technical punctuation,
versions, negation and code text with the model tokenizer. Negation and temporal
constraints remain retrieval failure cases to evaluate, not guaranteed constraints.

References: Manning [learning weights / Figure 6.5](https://nlp.stanford.edu/IR-book/html/htmledition/learning-weights-1.html), Chapters 13 and 14; class slides `Text Mining Part 1.pdf`,
PDF pages 3–5 and 13–20, inspected from the user's Drive course folder.
