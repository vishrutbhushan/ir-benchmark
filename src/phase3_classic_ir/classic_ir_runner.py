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
) -> dict[str, Any]:
    """
    Execute the full Phase 3 Classic IR pipeline.

    Args:
        tokenized_path: Path to outputs/phase2/tokenized_corpus.jsonl
        structured_path: Path to outputs/phase1/structured_records.jsonl
        queries_path: Path to queries.jsonl
        output_dir: Directory for Phase 3 outputs.
        logger: Logger instance.
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.
        top_k: Number of results per query.

    Returns:
        Summary statistics dict.
    """
    logger.info("Starting Phase 3 Classic IR Engine (BM25)...")
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
    write_json(index_stats, output_dir / "index_stats.json")
    logger.info(f"Index stats: {json.dumps(index_stats, indent=2)}")

    # ── Step 2: Initialize BM25 scorer ────────────────────────────────────
    scorer = BM25Scorer(index, k1=k1, b=b)
    logger.info(f"BM25 scorer initialized with k1={k1}, b={b}")

    # ── Step 3: Run query suite ───────────────────────────────────────────
    qp = QueryProcessor(index, scorer, logger)

    query_results: list[dict[str, Any]] = []
    if queries_path.exists():
        query_results = qp.run_query_suite(queries_path, top_k=top_k)

        # Save full search results
        results_path = output_dir / "bm25_results.json"
        write_json(query_results, results_path)
        logger.info(f"BM25 search results saved to {results_path}")
    else:
        logger.warning(f"Queries file not found at {queries_path}, skipping search.")

    # ── Step 4: Generate summary statistics ───────────────────────────────
    total_time = time.time() - start_time

    # Per-query stats
    query_summary: list[dict[str, Any]] = []
    for qr in query_results:
        top_result = qr["results"][0] if qr["results"] else None
        query_summary.append({
            "query_id": qr.get("query_id", ""),
            "query_category": qr.get("query_category", ""),
            "query_text": qr.get("query_text", ""),
            "num_tokens": len(qr.get("query_tokens", [])),
            "num_results": qr["num_results"],
            "top_score": top_result["score"] if top_result else 0,
            "top_doc_title": top_result["title"] if top_result else "",
            "top_doc_technology": top_result["technology"] if top_result else "",
        })

    summary = {
        "phase": "Phase 3 - Classic IR (BM25)",
        "parameters": {"k1": k1, "b": b, "top_k": top_k},
        "index_statistics": index_stats,
        "index_size_mb": round(index_size_mb, 2),
        "queries_executed": len(query_results),
        "query_summary": query_summary,
        "elapsed_seconds": round(total_time, 2),
    }

    write_json(summary, output_dir / "phase3_summary.json")

    # ── Step 5: Generate markdown report ──────────────────────────────────
    md_report = _generate_markdown_report(summary)
    write_text(md_report, output_dir / "phase3_summary.md")

    # ── Step 6: Generate charts ───────────────────────────────────────────
    _generate_charts(summary, query_results, index, charts_dir, logger)

    logger.info(f"Phase 3 Classic IR complete in {total_time:.1f}s. Outputs in {output_dir}")
    return summary


def _generate_markdown_report(summary: dict[str, Any]) -> str:
    """Generate markdown report of Phase 3 results."""
    lines = [
        "# Phase 3 - Classic IR Engine (BM25) Results\n",
        f"**BM25 Parameters:** k1={summary['parameters']['k1']}, b={summary['parameters']['b']}",
        f"**Top-K:** {summary['parameters']['top_k']}",
        f"**Index Size:** {summary['index_size_mb']} MB",
        f"**Elapsed Time:** {summary['elapsed_seconds']}s\n",
        "## Index Statistics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for key, val in summary["index_statistics"].items():
        if key != "avg_zone_lengths":
            lines.append(f"| {key} | {val:,} |" if isinstance(val, int) else f"| {key} | {val} |")

    avg_zl = summary["index_statistics"].get("avg_zone_lengths", {})
    if avg_zl:
        lines.append("")
        lines.append("## Average Zone Lengths\n")
        lines.append("| Zone | Avg Tokens |")
        lines.append("|------|-----------|")
        for zone, val in avg_zl.items():
            lines.append(f"| {zone.capitalize()} | {val} |")

    lines.append("")
    lines.append(f"## Query Results ({summary['queries_executed']} queries)\n")
    lines.append("| Query ID | Category | Top Score | Top Result | # Results |")
    lines.append("|----------|----------|-----------|------------|-----------|")

    for qs in summary.get("query_summary", []):
        title_short = qs["top_doc_title"][:50] + "..." if len(qs["top_doc_title"]) > 50 else qs["top_doc_title"]
        lines.append(
            f"| {qs['query_id']} | {qs['query_category']} | "
            f"{qs['top_score']:.4f} | {title_short} | {qs['num_results']} |"
        )

    return "\n".join(lines) + "\n"


def _generate_charts(
    summary: dict[str, Any],
    query_results: list[dict[str, Any]],
    index: InvertedIndex,
    charts_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate Phase 3 visualizations."""
    try:
        # Chart 1: Top BM25 scores per query
        if query_results:
            query_ids = [qr.get("query_id", f"Q{i}") for i, qr in enumerate(query_results)]
            top_scores = [
                qr["results"][0]["score"] if qr["results"] else 0
                for qr in query_results
            ]
            categories = [qr.get("query_category", "") for qr in query_results]

            # Color by query category
            cat_colors = {}
            color_palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                            "#00BCD4", "#8BC34A", "#FFC107", "#E91E63", "#673AB7"]
            unique_cats = list(set(categories))
            for i, cat in enumerate(unique_cats):
                cat_colors[cat] = color_palette[i % len(color_palette)]
            bar_colors = [cat_colors.get(c, "#999") for c in categories]

            fig, ax = plt.subplots(figsize=(14, 6))
            bars = ax.bar(query_ids, top_scores, color=bar_colors)
            ax.set_title("BM25 Top-1 Score per Query", fontsize=14, fontweight="bold")
            ax.set_xlabel("Query ID")
            ax.set_ylabel("BM25 Score")
            ax.tick_params(axis="x", rotation=45)

            # Legend
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=cat_colors[c]) for c in unique_cats
            ]
            ax.legend(legend_handles, unique_cats, loc="upper right", fontsize=8)

            plt.tight_layout()
            plt.savefig(charts_dir / "phase3_bm25_scores_per_query.png", dpi=150)
            plt.close()
            logger.info("Chart saved: phase3_bm25_scores_per_query.png")

        # Chart 2: Results count per query
        if query_results:
            result_counts = [qr["num_results"] for qr in query_results]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(query_ids, result_counts, color="#3F51B5")
            ax.set_title("Number of BM25 Results per Query", fontsize=14, fontweight="bold")
            ax.set_xlabel("Query ID")
            ax.set_ylabel("Documents Retrieved")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "phase3_results_count_per_query.png", dpi=150)
            plt.close()
            logger.info("Chart saved: phase3_results_count_per_query.png")

        # Chart 3: Document frequency distribution (top 30 terms)
        term_dfs = [(term, len(plist)) for term, plist in index.postings.items()]
        term_dfs.sort(key=lambda x: -x[1])
        top_terms = term_dfs[:30]
        if top_terms:
            terms_labels = [t[0][:15] for t in top_terms]
            terms_dfs = [t[1] for t in top_terms]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.barh(terms_labels[::-1], terms_dfs[::-1], color="#FF5722")
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
        ax.set_title("Unique Terms per Zone in Index", fontsize=14, fontweight="bold")
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
