# Neural architecture

```text
Prepared HTML → four fixed zones → overlapping token windows → stored embeddings
                                          ├─ relevance pairs → global zone weights
                                          └─ document vectors + content seeds → centroids
                                                                 ↓
                                                      corpus memberships + row lists

Query → embedding → nearest 3 centroids → category window rows → cosine
      → maximum per document/zone → weighted document score → document links
```

## Representation

The entire prepared corpus is retained: 68,414 documents and 476,906 windows. Collection, shared repairs and filtering belong to `data`. Neural text preserves case, negation, identifiers, versions and code indentation. Code is embedded as text; tables join body text. No code execution or language-specific parsing is used. Date mentions are not verified publication dates.

Four independent zones are title, headings, body and code. Default windows allow 496 content tokens with 64-token overlap and a checked 512-total-token limit. Each vector maps to a document, zone and window. Qwen3-Embedding-0.6B uses pinned revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, last-real-token pooling and L2 normalization, producing 1024 dimensions. Queries receive the retrieval instruction; document windows receive no synthetic prefixes.

SQLite holds text and metadata. FP16 NumPy arrays persist about 0.91 GiB of vectors; hand-coded FP32 cosine renormalizes stored vectors. No vector-search library or vector database handles ranking.

## Book-style zone weights

For each query/document pair, each feature is the highest window cosine in that zone. Missing zones contribute zero. The score is `sum(weight[zone] * cosine[zone])`, with nonnegative global weights summing to one. Similarities themselves are not normalized to sum to one.

Weights minimize squared error against binary relevance judgments using a deterministic simplex grid with step 0.01. This extends the book's Boolean-zone features to continuous cosine; the Boolean counting shortcut does not directly apply. There is no logistic loss, Precision@5 objective, or forced positive weight. Cosine is not a relevance probability, so this remains an approximation.

The reviewed development table has 240 pairs and 60 formulations across 20 intents: conversational, keywords and Boolean-style. Fifteen intents train weights; five validate them. Paraphrases stay together. Labels include rationales and source evidence offsets. Actual full-index cosine features are generated under `results/step_04_weights/features/`; label CSVs remain unchanged. These are small deliberately selected pools, not representative final evaluation. Consult the current weight report for learned values; a body-only solution is valid and does not demonstrate multi-zone improvement.

## Content classification and pruning

The 14 named classes are cloud platforms, AI/ML computation, MLOps, programming languages, databases/storage, search engines, messaging/RPC, distributed data processing, application frameworks, containers/orchestration, DevOps/build/release, OS/networking/web, security/identity and observability. Definitions and review decisions accompany the compact seed table.

There are 168 selected seeds: 10 training and 2 validation documents per class. Selection uses recognizable titles/URLs and source content; generic or misleading candidates are replaced. Primary labels describe content, not a technology-to-class mapping. Duplicate URL/content families do not cross splits. The small, deliberately clear validation set is a sanity check, not an estimate of query-routing relevance.

A document vector averages normalized window vectors within each zone, normalizes each available zone mean, then equally averages and normalizes the zone means. Thus longer zones do not gain influence merely through extra windows. Each class centroid averages its training document vectors and is normalized. Validation documents never fit centroids.

Every corpus document receives its highest-cosine class; a second is added when within 0.03 of the best score. This configurable margin is a heuristic, not a probability. Centroids stay fixed during assignment. Category lists store row IDs into the existing zone matrices, not duplicate vectors.

Queries select the nearest 3 available classes by default. Their window lists are unioned/deduplicated, filtered if requested, and scored. Category selection prunes; zones rank. This is a local partitioned index, not distributed search or RAPTOR. Pruning can exclude relevant documents, and there is no retention guarantee. No NB model or summary hierarchy remains.

## Search and configuration

Modes are exhaustive `dense` (maximum across all windows), exhaustive `zones`, and `routed-centroid`. Ties use document ID, then earliest winning window. Results show document titles, technology, assigned content categories and numerical scores. Main links open sandboxed HTML from the corpus ZIP or extracted folder; secondary links open original websites. The UI has no technology/category filters or query suggestions. Cached pagination shows five results per page from the configured top ten. Boolean-looking queries are semantic inputs, not executable Boolean expressions.

`config.json` controls the pinned model/revision/weight checksum, token limits, overlap, device, batch size, workers, weighting/routing switches, routing depth, membership margin cache limits, result_limit and page_size. Alternate models must satisfy the implemented Qwen3 pooling and single-weight-file contract and provide their own revision/checksum. CLI run options override defaults. Model/window changes require compatible new embeddings; label changes require refitting derived learning artifacts. Existing completed embeddings/checkpoints are preserved when compatible.

Step 07 verifies the full index and development-query retrieval, including cosine agreement, mappings, coverage, result limits and pagination. Learning audits check references, evidence offsets, features and split consistency. Final challenge queries, relevance metrics and the classic/neural comparison belong to `evaluation` and are not run here.

Run logs record stage times, progress and tracebacks under `logs/`. Keep only the current model, vectors, passage databases and learned artifacts in `outputs/`. The pipeline does not copy source documents into learning folders or write a duplicate extracted JSONL file.

The current squared-error fit chooses title/headings/body/code weights 0/0/1/0. Training error is 0.109683 for body alone, 0.113208 for 10% title plus 90% body, and 0.140820 for equal weights. This describes the best fit on the selected examples; it does not establish that other zones contain no useful information.
