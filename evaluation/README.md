# Retrieval evaluation

This branch contains fixed results and a labeling UI. It does not retrieve or embed documents.

## Run

Run these commands from an `evaluation` branch checkout. Python 3 is enough; no package installation is required.

```bash
python3 run_evaluation.py
```

To enable saved HTML links, supply the corpus ZIP:

```bash
python3 run_evaluation.py --corpus-zip /path/to/technical_web_corpus.zip
```

From the existing neural checkout with the local evaluation worktree:

```bash
.venv/bin/python .worktrees/evaluation/run_evaluation.py --corpus-zip data/00_corpus/technical_web_corpus.zip --port 8001
```

Stop with Ctrl+C. Judgments remain saved when you restart.

Open **http://127.0.0.1:8001**. Choose Classical (BM25) or Neural (dense cosine), and judge one query at a time. Relevant and Non-relevant buttons save immediately to `data/judgments.csv`. Clear removes a judgment. Blank means unjudged. Labels carry across variants for the same query and document.

There are exactly 20 queries and 10 results per variant. The two runs contain 400 result rows but only 344 unique query/document pairs to judge. Original links open in a new tab. The supplied ZIP enables saved HTML links where the document exists.

## Results and exports

- `queries/final_queries.jsonl`: the frozen 20 queries with original IDs and text.
- `queries/selection.csv`: decisions for all 34 source candidates, made for specificity and coverage.
- `results/classical/` and `results/neural/`: matching CSV schemas, timing CSVs and provenance manifests.
- `config/neural_variants.json`: the single active dense baseline.
- `results/comparison.json`: corpus mismatch and missing archived documents.
- UI downloads: judgments, results with labels, metrics, per-query metrics, TREC qrels and timings.

The active comparison is BM25 versus plain dense cosine retrieval, matching the course brief. Manual zone weights and category routing are not used in this neural baseline. Other generated runs remain in the local ignored archive and do not increase the judgment pool.

```bash
python run_evaluation.py --metrics
```

This writes `results/metrics.csv` and `results/per_query.csv`. Precision is available after each variant's returned documents are judged. Recall, F1, AP and NDCG wait until the full query pool is judged. Unready values remain blank; zero is a measured result, not a missing label.

Metrics use K=1,3,5,10. Recall and F1 use all relevant documents in the combined exported result pool. AP@10 divides the sum of precision at relevant ranks by the number of relevant pooled documents. MAP@10 averages those AP values across all 20 queries. Binary NDCG uses gain 0/1 and log2(rank+1) discount. Queries with no relevant pooled documents score zero and remain visible. MRR@10 is included. Full-corpus recall and untruncated MAP cannot be inferred from these top-ten pools.

## Comparison limits

Use the supplied exports for labeling now. The manifests retain the original corpus counts and provenance; differing corpora can affect document statistics and rankings. Do not remove unavailable classical hits or invent replacement rankings.

The classical files lack per-query timing, RSS, startup time, exact index bytes, corpus manifest hash and cache policy. Those fields remain blank. Their legacy aggregate times cover 34 queries and top 20, so they are not comparable to this neural run. Neural timings include uncached query encoding, ranking and sorting over a warm model, with three repetitions. RSS high-water marks cover the process lifetime; CUDA peaks include the resident model and index. Full index-build time was not measured in this run.

The supplied query suite covers negation, typos and a version constraint. It still lacks an explicit homonym-disambiguation pair and before/after temporal queries from the original course brief. New queries require new classical snapshots as well.

To regenerate snapshots, import classical data read-only with `--import-classical /path/to/repo --revision COMMIT`. Generate neural CSVs from the neural branch with its pipeline's `--step export`, using this branch's query and variant files. No corpus ZIP, model weights or embeddings belong in Git. No tests are included.
