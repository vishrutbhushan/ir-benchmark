"""
Classic IR Runner — Phase 3 pipeline entry point.

Orchestrates:
    1. Building the inverted index from tokenized corpus
    2. Running BM25 search on the query suite
    3. Generating statistics and visualizations
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.phase3_classic_ir.inverted_index import InvertedIndex
from src.phase3_classic_ir.bm25_scorer import BM25Scorer
from src.phase3_classic_ir.bm25f_scorer import BM25FScorer, DEFAULT_ZONE_WEIGHTS
from src.phase3_classic_ir.inverted_index import InvertedIndex
from src.phase3_classic_ir.query_processor import QueryProcessor
from src.utils.file_utils import write_json, write_text


def run_classic_ir(
    tokenized_path: Path,
    structured_path: Path,
    queries_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    k1: float = 1.2,
    b: float = 0.75,
    top_k: int = 20,
    scorer_type: str = "both",
    zone_weights: dict[str, float] | None = None,
    hierarchical_boost: float = 0.25,
    routing_mode: str = "hybrid",
) -> dict[str, Any]:
    """
    Execute the full Phase 3 Classic IR pipeline supporting Flat BM25,
    Hierarchical Zone-Weighted BM25F, and Multi-Level Section/Passage Retrieval.

    Args:
        tokenized_path: Path to outputs/phase2/tokenized_corpus.jsonl
        structured_path: Path to outputs/phase1/structured_records.jsonl
        queries_path: Path to queries.jsonl
        output_dir: Directory for Phase 3 outputs.
        logger: Logger instance.
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
        top_k: Number of results per query.
        scorer_type: "bm25", "bm25f", or "both".
        zone_weights: Custom zone weights for BM25F.
        hierarchical_boost: Taxonomy matching bonus factor for BM25F.
        routing_mode: "hybrid" (soft boost), "hierarchical_first" (category sub-index routing),
                      or "tiered" (champion list candidate pruning).

    Returns:
        Summary statistics dict.
    """
    logger.info(f"Starting Phase 3 Classic IR Engine [Scorer: {scorer_type}, Routing: {routing_mode}]...")
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # ── Step 1: Build inverted index ──────────────────────────────────────
    index = InvertedIndex()
    index.build_from_tokenized_corpus(tokenized_path, structured_path, logger)

    # Save index to disk
    index_path = output_dir / "inverted_index.json"
    index.save(index_path)
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    logger.info(f"Inverted index saved to {index_path} ({index_size_mb:.1f} MB)")

    # Save index statistics
    index_stats = index.get_index_stats()
    index_stats["hierarchical_categories"] = len(index.category_postings)
    index_stats["hierarchical_technologies"] = len(index.technology_postings)
    index_stats["hierarchical_domains"] = len(index.domain_postings)
    index_stats["total_sections_indexed"] = len(index.section_lengths)
    index_stats["avg_section_length"] = round(index.avg_section_length, 1)
    index_stats["champion_lists_count"] = len(index.champion_postings)
    write_json(index_stats, output_dir / "index_stats.json")
    logger.info(f"Index stats: {json.dumps(index_stats, indent=2)}")

    # ── Step 2: Run Scorers ───────────────────────────────────────────────
    bm25_results: list[dict[str, Any]] = []
    bm25f_results: list[dict[str, Any]] = []
    multi_level_results: list[dict[str, Any]] = []

    if queries_path.exists():
        # 1. Flat BM25
        if scorer_type in ["bm25", "both"]:
            logger.info(f"Executing Flat BM25 retrieval (k1={k1}, b={b})...")
            bm25_scorer = BM25Scorer(index, k1=k1, b=b)
            qp_bm25 = QueryProcessor(index, bm25_scorer, logger)
            bm25_results = qp_bm25.run_query_suite(queries_path, top_k=top_k)
            write_json(bm25_results, output_dir / "bm25_results.json")
            logger.info("Saved Flat BM25 results to bm25_results.json")

        # 2. Hierarchical BM25F
        if scorer_type in ["bm25f", "both"]:
            active_weights = zone_weights or DEFAULT_ZONE_WEIGHTS
            logger.info(
                f"Executing Hierarchical Zone-Weighted BM25F retrieval (weights={active_weights}, boost={hierarchical_boost}, routing={routing_mode})..."
            )
            bm25f_scorer = BM25FScorer(
                index,
                k1=k1,
                zone_weights=active_weights,
                hierarchical_boost=hierarchical_boost,
            )
            qp_bm25f = QueryProcessor(index, bm25f_scorer, logger, routing_mode=routing_mode)
            bm25f_results = qp_bm25f.run_query_suite(queries_path, top_k=top_k)
            write_json(bm25f_results, output_dir / "bm25f_results.json")
            logger.info("Saved Hierarchical BM25F results to bm25f_results.json")

            # 3. Multi-Level Document -> Section / Passage Retrieval Suite
            logger.info("Executing Multi-Level Document & Section/Passage retrieval suite...")
            with queries_path.open("r", encoding="utf-8") as f:
                for line in f:
                    qrec = json.loads(line)
                    qid = qrec["query_id"]
                    qtext = qrec["query_text"]
                    qtokens = qp_bm25f.process_query(qtext)
                    ml_res = bm25f_scorer.score_query_multi_level(
                        query_terms=qtokens,
                        query_raw_text=qtext,
                        top_k_docs=top_k,
                        top_k_sections=5,
                        routing_mode=routing_mode,
                    )
                    multi_level_results.append({
                        "query_id": qid,
                        "query_text": qtext,
                        "query_category": qrec.get("category", ""),
                        "documents": ml_res["documents"],
                        "sections": ml_res["sections"],
                    })
            write_json(multi_level_results, output_dir / "multi_level_results.json")
            logger.info("Saved Multi-Level retrieval results to multi_level_results.json")
    else:
        logger.warning(f"Queries file not found at {queries_path}, skipping search.")

    # ── Step 3: Comparative Analysis ─────────────────────────────────────
    total_time = time.time() - start_time
    primary_results = bm25f_results if scorer_type == "bm25f" else bm25_results
    if not primary_results and bm25f_results:
        primary_results = bm25f_results

    comparison_summary: list[dict[str, Any]] = []
    for i, qr in enumerate(primary_results):
        qid = qr.get("query_id", f"Q{i+1}")
        qcat = qr.get("query_category", "")
        qtext = qr.get("query_text", "")

        bm25_top = bm25_results[i]["results"][0] if i < len(bm25_results) and bm25_results[i]["results"] else None
        bm25f_top = bm25f_results[i]["results"][0] if i < len(bm25f_results) and bm25f_results[i]["results"] else None

        comparison_summary.append({
            "query_id": qid,
            "query_category": qcat,
            "query_text": qtext,
            "bm25_top_score": bm25_top["score"] if bm25_top else 0.0,
            "bm25_top_doc": bm25_top["title"] if bm25_top else "",
            "bm25_top_tech": bm25_top["technology"] if bm25_top else "",
            "bm25f_top_score": bm25f_top["score"] if bm25f_top else 0.0,
            "bm25f_top_doc": bm25f_top["title"] if bm25f_top else "",
            "bm25f_top_tech": bm25f_top["technology"] if bm25f_top else "",
            "bm25f_top_category": bm25f_top.get("category", "") if bm25f_top else "",
        })

    summary = {
        "phase": "Phase 3 - Classic IR (BM25 & Hierarchical BM25F)",
        "scorer_type": scorer_type,
        "parameters": {
            "k1": k1,
            "b": b,
            "top_k": top_k,
            "hierarchical_boost": hierarchical_boost,
            "zone_weights": zone_weights or DEFAULT_ZONE_WEIGHTS,
        },
        "index_statistics": index_stats,
        "index_size_mb": round(index_size_mb, 2),
        "queries_executed": len(primary_results),
        "comparison_summary": comparison_summary,
        "elapsed_seconds": round(total_time, 2),
    }

    write_json(summary, output_dir / "phase3_summary.json")

    # ── Step 4: Generate Markdown Report & Charts ─────────────────────────
    md_report = _generate_markdown_report(summary, bm25_results, bm25f_results)
    write_text(md_report, output_dir / "phase3_summary.md")

    _generate_charts(summary, bm25_results, bm25f_results, index, charts_dir, logger)

    logger.info(f"Phase 3 Classic IR complete in {total_time:.1f}s. Outputs in {output_dir}")
    return summary


def _generate_markdown_report(
    summary: dict[str, Any],
    bm25_results: list[dict[str, Any]],
    bm25f_results: list[dict[str, Any]],
) -> str:
    """Generate markdown report comparing BM25 and Hierarchical BM25F results."""
    params = summary["parameters"]
    lines = [
        "# Phase 3 — Classic IR Engine: BM25 vs. Hierarchical BM25F Results\n",
        f"**Engine Configuration:** Scorer Mode: `{summary.get('scorer_type', 'both')}`",
        f"**BM25 Parameters:** k1={params['k1']}, b={params['b']}, top_k={params['top_k']}",
        f"**BM25F Zone Weights:** {params.get('zone_weights', {})}",
        f"**Hierarchical Taxonomy Boost:** {params.get('hierarchical_boost', 0.25) * 100:.0f}%",
        f"**Index Size:** {summary['index_size_mb']} MB",
        f"**Elapsed Time:** {summary['elapsed_seconds']}s\n",
        "## 1. Index Statistics & Hierarchical Postings\n",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    for key, val in summary["index_statistics"].items():
        if key != "avg_zone_lengths":
            lines.append(f"| {key.replace('_', ' ').title()} | {val:,} |" if isinstance(val, int) else f"| {key} | {val} |")

    avg_zl = summary["index_statistics"].get("avg_zone_lengths", {})
    if avg_zl:
        lines.append("")
        lines.append("## 2. Average Zone Lengths across Corpus\n")
        lines.append("| Zone | Average Tokens |")
        lines.append("| :--- | :--- |")
        for zone, val in avg_zl.items():
            lines.append(f"| **{zone.capitalize()}** | {val:.1f} tokens |")

    lines.append("")
    lines.append("## 3. Query Suite Comparative Results (BM25 vs. Hierarchical BM25F)\n")
    lines.append("| Query ID | Category | Query Text | BM25 Top Score | BM25F Top Score | BM25F Top Tech / Domain |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for item in summary.get("comparison_summary", []):
        qtext_short = item["query_text"][:45] + "..." if len(item["query_text"]) > 45 else item["query_text"]
        tech_cat = f"{item['bm25f_top_tech']} ({item['bm25f_top_category']})" if item['bm25f_top_tech'] else "—"
        lines.append(
            f"| **{item['query_id']}** | {item['query_category']} | {qtext_short} | "
            f"`{item['bm25_top_score']:.4f}` | **`{item['bm25f_top_score']:.4f}`** | {tech_cat} |"
        )

    return "\n".join(lines) + "\n"


def _generate_charts(
    summary: dict[str, Any],
    bm25_results: list[dict[str, Any]],
    bm25f_results: list[dict[str, Any]],
    index: InvertedIndex,
    charts_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate Phase 3 comparative and diagnostic visualizations."""
    try:
        import numpy as np

        # Chart 1: Side-by-side Top-1 Score Comparison: BM25 vs Hierarchical BM25F
        comparison = summary.get("comparison_summary", [])
        if comparison:
            query_ids = [c["query_id"] for c in comparison]
            bm25_scores = [c["bm25_top_score"] for c in comparison]
            bm25f_scores = [c["bm25f_top_score"] for c in comparison]

            x = np.arange(len(query_ids))
            width = 0.38

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.bar(x - width/2, bm25_scores, width, label="Flat BM25", color="#2196F3", alpha=0.85)
            ax.bar(x + width/2, bm25f_scores, width, label="Hierarchical BM25F", color="#4CAF50", alpha=0.95)

            ax.set_title("BM25 vs. Hierarchical BM25F Top-1 Scores Across Queries", fontsize=14, fontweight="bold")
            ax.set_xlabel("Query ID", fontweight="bold")
            ax.set_ylabel("Top Retrieval Score", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(query_ids, rotation=45, ha="right")
            ax.legend(fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            plt.tight_layout()
            plt.savefig(charts_dir / "phase3_bm25_vs_bm25f_scores.png", dpi=150)
            plt.close()
            logger.info("Chart saved: phase3_bm25_vs_bm25f_scores.png")

        # Chart 2: Document count per Category in Hierarchical Index
        if index.category_postings:
            cat_names = sorted(list(index.category_postings.keys()))
            cat_counts = [len(index.category_postings[c]) for c in cat_names]

            fig, ax = plt.subplots(figsize=(13, 6))
            ax.barh(cat_names[::-1], cat_counts[::-1], color="#FF9800", alpha=0.85)
            ax.set_title("Documents Indexed per Category (Hierarchical Index)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Indexed Documents", fontweight="bold")
            for i, v in enumerate(cat_counts[::-1]):
                ax.text(v + max(1, max(cat_counts) * 0.01), i, f"{v:,}", va="center", fontsize=9)

            plt.tight_layout()
            plt.savefig(charts_dir / "phase3_hierarchical_category_distribution.png", dpi=150)
            plt.close()
            logger.info("Chart saved: phase3_hierarchical_category_distribution.png")

        # Chart 3: Document frequency distribution (top 30 terms)
        term_dfs = [(term, len(plist)) for term, plist in index.postings.items()]
        term_dfs.sort(key=lambda x: -x[1])
        top_terms = term_dfs[:30]
        if top_terms:
            terms_labels = [t[0][:15] for t in top_terms]
            terms_dfs = [t[1] for t in top_terms]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.barh(terms_labels[::-1], terms_dfs[::-1], color="#E91E63", alpha=0.85)
            ax.set_title("Top 30 Terms by Document Frequency", fontsize=14, fontweight="bold")
            ax.set_xlabel("Document Frequency (# docs containing term)")
            plt.tight_layout()
            plt.savefig(charts_dir / "phase3_top_terms_by_df.png", dpi=150)
            plt.close()
            logger.info("Chart saved: phase3_top_terms_by_df.png")

        # Chart 4: Vocabulary coverage per zone
        zone_vocabs: dict[str, set[str]] = defaultdict(set)
        for term, plist in index.postings.items():
            for posting in plist:
                for zone, tf in posting.zone_tfs.items():
                    if tf > 0:
                        zone_vocabs[zone].add(term)

        zones = list(zone_vocabs.keys())
        vocab_sizes = [len(zone_vocabs[z]) for z in zones]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            [z.capitalize() for z in zones], vocab_sizes,
            color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        )
        ax.set_title("Unique Terms per Zone in Inverted Index", fontsize=14, fontweight="bold")
        ax.set_ylabel("Vocabulary Size")
        for bar, val in zip(bars, vocab_sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(charts_dir / "phase3_vocab_per_zone_index.png", dpi=150)
        plt.close()
        logger.info("Chart saved: phase3_vocab_per_zone_index.png")

    except Exception as e:
        logger.error(f"Error generating Phase 3 charts: {e}")
