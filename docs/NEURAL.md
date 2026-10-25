# Neural architecture

The input is the corrected corpus ZIP from `data`. This branch does not collect pages or apply shared corpus filtering.

```text
01 Prepared HTML → structured SQLite records
02 Four fixed zones → overlapping token windows
03 Pinned Qwen3 encoder → four persistent vector matrices
04 Relevance labels → learned zone weights
05 Source-domain labels → NB and neural centroid models
06 Development queries → validated class-routing depth
07 Full-index and learning-data verification
08 Query embedding → cosine → document scores → ranking
09 Search UI → filters, snippets and cached pagination
```

## Inputs and representation

The current snapshot has 68,414 documents across 76 technologies. Extracted records retain document IDs, URLs, titles, headings, sections, body, code, date mentions and metadata. Neural text preserves negation, case, punctuation, versions and identifiers. Code keeps indentation and uses the same text encoder. Tables become visible body text. Dates are mentions, not verified publication dates.

Title, headings, body and code are separate zones. Each uses up to 496 content tokens, 64-token overlap and a checked 512-total-token limit. The index has 68,414 title, 61,511 heading, 255,445 body and 91,536 code windows.

Qwen3-Embedding-0.6B is pinned to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Last-real-token pooling and L2 normalization produce 1,024-dimensional vectors. Queries receive the model's retrieval instruction. Documents do not receive synthetic context prefixes. No model fine-tuning is performed.

SQLite stores text and metadata. NumPy FP16 matrices persist about 0.91 GiB of vectors. Scoring uses FP32 cosine with renormalization and direct matrix operations, optionally GPU-resident. Completed embedding checkpoints are durable and reusable.

## Ranking and learning

`dense` takes the highest window cosine per document across all zones. `zones` combines per-zone maximum cosines using learned nonnegative weights, renormalized over present zones. If all present-zone weights are zero, it uses equal weights for those zones. Ties use document ID; winning-window ties use the earliest index.

Weight fitting uses 262 binary relevance pairs across 44 development queries (33 training / 11 validation queries). A deterministic simplex grid with step 0.05 minimizes training squared error. Current title/headings/body/code weights are **0/0/1/0**; this does not demonstrate a benefit from blending zones. Cosines are features; relevance labels remain 0/1.

Classification uses 14 fixed source-domain classes. The 420-document table contains only doc_id, class_id, class_name and split: 20/5/5 train/validation/test per class. Full text is loaded by ID; duplicate families stay within one split. Readable source files are in each learning folder's documents/ directory.

NB uses training-only word counts, vocabulary with df ≥ 2, add-one smoothing and empirical priors. Neural centroids average normalized training-document vectors; document vectors average their window embeddings. Test accuracy is 74.3% for NB and 84.3% for centroids, measuring source-label agreement rather than relevance.

`routed-nb` and `routed-centroid` select classes before weighted cosine retrieval. Routing chooses the smallest top-k retaining mean 98% of exhaustive weighted top-20 results and at least 90% per validation query. Current k values are 12/14 and 13/14, so pruning benefit is limited. This is supervised source-class routing, not clustering or RAPTOR; the summary hierarchy is removed.

## Checks and branch boundary

Step 07 checks every window's source offsets, vector/document mappings, normalization, GPU/NumPy cosine agreement, development-query ranking, filters and cached pagination. These are engineering checks, not relevance evaluation.

Final challenge queries, qrels, NDCG/MAP comparison and failure analysis belong to `evaluation`. Development queries remain here because they are required for learning. The book connections retained in this branch are vector-space cosine, learned weighted zones, Naive Bayes and Rocchio-style classification.

Small analytical reports live in numbered results/ folders and may be versioned. Large runtime artifacts stay under ignored outputs/. No vector database or delegated ranking library is used.
