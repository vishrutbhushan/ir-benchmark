#!/usr/bin/env python3
"""
Benchmark Runner for Test Queries (NL, Short Keywords, Boolean, Exact Code, Typos).
Loads the pre-built Phase 3 index and outputs retrieval results to outputs/test_queries_run/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.phase3_classic_ir.inverted_index import InvertedIndex
from src.phase3_classic_ir.bm25_scorer import BM25Scorer
from src.phase3_classic_ir.bm25f_scorer import BM25FScorer, DEFAULT_ZONE_WEIGHTS
from src.phase3_classic_ir.query_processor import QueryProcessor
from src.phase3_classic_ir.multilevel_index import MultiLevelIndex
from src.utils.file_utils import write_json, write_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_queries_runner")


def run_benchmark():
    test_queries_path = Path("test_queries.jsonl")
    output_dir = Path("outputs/test_queries_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = Path("outputs/phase3/inverted_index.json")
    catalog_path = Path("outputs/phase3/document_catalog.json")

    logger.info("Loading pre-built inverted index and catalog...")
    t0 = time.time()
    index = InvertedIndex.load(index_path, catalog_path)
    logger.info(f"Loaded index ({len(index.vocabulary):,} vocabulary terms, {index.total_docs:,} docs) in {time.time()-t0:.1f}s")

    # Read all test queries
    queries: list[dict[str, Any]] = []
    with test_queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    logger.info(f"Loaded {len(queries)} test queries from {test_queries_path}")

    # 1. Run Flat BM25
    logger.info("Running Model 1: Flat BM25...")
    bm25_scorer = BM25Scorer(index, k1=1.2, b=0.75)
    qp_bm25 = QueryProcessor(index, bm25_scorer, logger)

    bm25_results = []
    t_start_bm25 = time.time()
    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        res = qp_bm25.search(qtext, top_k=20)
        res["query_id"] = qid
        res["query_type"] = q.get("query_type", "")
        res["category"] = q.get("category", "")
        res["target_tech"] = q.get("target_tech", "")
        bm25_results.append(res)
    bm25_time = time.time() - t_start_bm25
    write_json(bm25_results, output_dir / "bm25_results.json")
    logger.info(f"Flat BM25 completed in {bm25_time:.2f}s ({bm25_time/len(queries)*1000:.1f}ms/query)")

    # 2. Run Zonal BM25F
    logger.info("Running Model 2: Zone-Weighted BM25F...")
    bm25f_scorer = BM25FScorer(index, k1=1.2, zone_weights=DEFAULT_ZONE_WEIGHTS, hierarchical_boost=0.25)
    qp_bm25f = QueryProcessor(index, bm25f_scorer, logger, routing_mode="hybrid")

    bm25f_results = []
    t_start_bm25f = time.time()
    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        res = qp_bm25f.search(qtext, top_k=20)
        res["query_id"] = qid
        res["query_type"] = q.get("query_type", "")
        res["category"] = q.get("category", "")
        res["target_tech"] = q.get("target_tech", "")
        bm25f_results.append(res)
    bm25f_time = time.time() - t_start_bm25f
    write_json(bm25f_results, output_dir / "bm25f_results.json")
    logger.info(f"Zone BM25F completed in {bm25f_time:.2f}s ({bm25f_time/len(queries)*1000:.1f}ms/query)")

    # 3. Run Multi-Level Indexing (Document + Best Section Snippet)
    logger.info("Running Model 3: Multi-Level Document & Section/Passage retrieval...")
    ml_index = MultiLevelIndex(index)
    ml_results = []
    t_start_ml = time.time()
    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        tokens = qp_bm25f.process_query(qtext)
        doc_sections = ml_index.search_multilevel(
            query_terms=tokens,
            query_raw_text=qtext,
            top_k_docs=20,
            top_k_sections_per_doc=1,
        )
        ml_results.append({
            "query_id": qid,
            "query_text": qtext,
            "query_type": q.get("query_type", ""),
            "category": q.get("category", ""),
            "target_tech": q.get("target_tech", ""),
            "tokens": tokens,
            "num_results": len(doc_sections),
            "results": doc_sections,
        })
    ml_time = time.time() - t_start_ml
    write_json(ml_results, output_dir / "multilevel_results.json")
    logger.info(f"Multi-Level completed in {ml_time:.2f}s ({ml_time/len(queries)*1000:.1f}ms/query)")

    # 4. Generate Comparative Summary Markdown
    logger.info("Generating comparative markdown summary...")
    lines = [
        "# Test Queries Benchmark — Classical IR Retrieval Results",
        "",
        f"- **Queries Evaluated**: {len(queries)} across 5 Query Types (Natural Language, Short Keyword, Boolean, Exact Terminology, Spelling Errors)",
        f"- **Index Documents**: {index.total_docs:,}",
        f"- **Index Vocabulary**: {len(index.vocabulary):,} unique stemmed terms",
        f"- **BM25 Execution Time**: {bm25_time:.2f}s ({bm25_time/len(queries)*1000:.1f}ms/q)",
        f"- **BM25F Execution Time**: {bm25f_time:.2f}s ({bm25f_time/len(queries)*1000:.1f}ms/q)",
        f"- **Multi-Level Execution Time**: {ml_time:.2f}s ({ml_time/len(queries)*1000:.1f}ms/q)",
        "",
        "## Performance By Query Type & Category",
        "",
        "| Query ID | Type | Target Tech | Query Text | BM25 Top Score | BM25F Top Score | Top Retrieved Tech (BM25F) | Top Match Zone/Title |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for b25, b25f, ml in zip(bm25_results, bm25f_results, ml_results):
        qid = b25["query_id"]
        qtype = b25.get("query_type", "")
        target = b25.get("target_tech", "")
        qtxt = b25["query_text"]
        if len(qtxt) > 42:
            qtxt = qtxt[:39] + "..."
        
        top_b25 = b25["results"][0] if b25["results"] else {}
        top_b25f = b25f["results"][0] if b25f["results"] else {}
        
        b25_score = f"{top_b25.get('score', 0):.2f}" if top_b25 else "0.00"
        b25f_score = f"{top_b25f.get('score', 0):.2f}" if top_b25f else "0.00"
        ret_tech = top_b25f.get("technology", "None")
        title = top_b25f.get("title", "")
        if len(title) > 35:
            title = title[:32] + "..."

        lines.append(f"| **{qid}** | `{qtype}` | {target} | {qtxt} | {b25_score} | **{b25f_score}** | {ret_tech} | {title} |")

    lines.append("")
    summary_path = output_dir / "test_queries_summary.md"
    write_text("\n".join(lines), summary_path)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run_benchmark()
