"""
Tokenization Runner — Phase 2 pipeline entry point.

Reads structured records from Phase 1, tokenizes all zones using the shared
tokenizer, produces tokenized corpus output and statistics report with
visualizations.
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

from src.phase2_tokenization.tokenizer import tokenize_zones
from src.utils.file_utils import stream_jsonl, write_json, write_text


def run_tokenization(
    records_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    max_documents: int | None = None,
) -> dict[str, Any]:
    """
    Execute Phase 2 tokenization pipeline.

    Reads structured_records.jsonl from Phase 1, tokenizes every zone
    of every document, and writes:
        - tokenized_corpus.jsonl  (per-doc zone token lists)
        - tokenization_stats.json (statistics summary)
        - tokenization_stats.md   (markdown report)
        - charts/                 (visualizations)

    Args:
        records_path: Path to Phase 1 structured_records.jsonl.
        output_dir: Directory for Phase 2 outputs.
        logger: Logger instance.
        max_documents: Optional limit for testing.

    Returns:
        Statistics dictionary.
    """
    logger.info("Starting Phase 2 Tokenization...")
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    tokenized_path = output_dir / "tokenized_corpus.jsonl"

    # Accumulators for statistics
    doc_count = 0
    total_tokens_by_zone: Counter[str] = Counter()
    zone_doc_counts: Counter[str] = Counter()  # docs with ≥1 token per zone
    total_token_count = 0
    tokens_per_doc: list[int] = []
    category_token_counts: defaultdict[str, int] = defaultdict(int)
    category_doc_counts: defaultdict[str, int] = defaultdict(int)
    vocab_by_zone: dict[str, set[str]] = {
        "title": set(), "heading": set(), "body": set(),
        "code": set(), "section": set(),
    }

    start_time = time.time()

    with tokenized_path.open("w", encoding="utf-8") as out_f:
        for record in stream_jsonl(records_path):
            if max_documents is not None and doc_count >= max_documents:
                break

            tokenized = tokenize_zones(record)
            out_f.write(json.dumps(tokenized, ensure_ascii=False) + "\n")

            doc_count += 1
            doc_total = tokenized["total_token_count"]
            total_token_count += doc_total
            tokens_per_doc.append(doc_total)

            # Per-zone stats
            for zone_key in ["title", "heading", "body", "code", "section"]:
                tokens_key = f"{zone_key}_tokens"
                zone_tokens = tokenized.get(tokens_key, [])
                zone_len = len(zone_tokens)
                total_tokens_by_zone[zone_key] += zone_len
                if zone_len > 0:
                    zone_doc_counts[zone_key] += 1
                vocab_by_zone[zone_key].update(zone_tokens)

            # Per-category stats
            category = record.get("metadata", {}).get("category", "unknown")
            category_token_counts[category] += doc_total
            category_doc_counts[category] += 1

            if doc_count % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Tokenized {doc_count:,} docs ({elapsed:.1f}s, "
                    f"{doc_count / elapsed:.1f} docs/s)..."
                )

    total_time = time.time() - start_time
    logger.info(
        f"Tokenization finished in {total_time:.1f}s. "
        f"Processed {doc_count:,} documents, {total_token_count:,} total tokens."
    )

    # ── Compute statistics ────────────────────────────────────────────────
    avg_tokens = total_token_count / doc_count if doc_count else 0
    sorted_per_doc = sorted(tokens_per_doc)
    median_tokens = (
        sorted_per_doc[len(sorted_per_doc) // 2] if sorted_per_doc else 0
    )

    vocab_sizes = {zone: len(vocab) for zone, vocab in vocab_by_zone.items()}
    total_vocab = len(set().union(*vocab_by_zone.values())) if vocab_by_zone else 0

    zone_stats = {}
    for zone_key in ["title", "heading", "body", "code", "section"]:
        zone_total = total_tokens_by_zone[zone_key]
        zone_docs = zone_doc_counts[zone_key]
        zone_stats[zone_key] = {
            "total_tokens": zone_total,
            "docs_with_tokens": zone_docs,
            "avg_tokens_per_doc": round(zone_total / zone_docs, 1) if zone_docs else 0,
            "vocabulary_size": vocab_sizes[zone_key],
        }

    category_stats = {}
    for cat, count in sorted(category_doc_counts.items()):
        cat_tokens = category_token_counts[cat]
        category_stats[cat] = {
            "doc_count": count,
            "total_tokens": cat_tokens,
            "avg_tokens_per_doc": round(cat_tokens / count, 1) if count else 0,
        }

    stats = {
        "total_documents": doc_count,
        "total_tokens": total_token_count,
        "average_tokens_per_doc": round(avg_tokens, 1),
        "median_tokens_per_doc": median_tokens,
        "total_vocabulary_size": total_vocab,
        "zone_statistics": zone_stats,
        "category_statistics": category_stats,
        "elapsed_seconds": round(total_time, 2),
    }

    # ── Write outputs ─────────────────────────────────────────────────────
    write_json(stats, output_dir / "tokenization_stats.json")
    logger.info(f"Saved tokenization statistics JSON to {output_dir / 'tokenization_stats.json'}")

    md_report = _generate_markdown_report(stats)
    write_text(md_report, output_dir / "tokenization_stats.md")
    logger.info(f"Saved tokenization statistics MD to {output_dir / 'tokenization_stats.md'}")

    # ── Generate charts ───────────────────────────────────────────────────
    _generate_charts(stats, tokens_per_doc, charts_dir, logger)

    logger.info(f"Phase 2 tokenization complete. All outputs in {output_dir}")
    return stats


def _generate_markdown_report(stats: dict[str, Any]) -> str:
    """Generate a markdown summary of tokenization statistics."""
    lines = [
        "# Phase 2 — Tokenization Statistics\n",
        f"**Documents tokenized:** {stats['total_documents']:,}",
        f"**Total tokens:** {stats['total_tokens']:,}",
        f"**Average tokens/doc:** {stats['average_tokens_per_doc']:,.1f}",
        f"**Median tokens/doc:** {stats['median_tokens_per_doc']:,}",
        f"**Total vocabulary size:** {stats['total_vocabulary_size']:,}\n",
        "## Per-Zone Statistics\n",
        "| Zone | Total Tokens | Docs w/ Tokens | Avg/Doc | Vocab Size |",
        "|------|-------------|----------------|---------|------------|",
    ]
    for zone, zs in stats["zone_statistics"].items():
        lines.append(
            f"| {zone.capitalize()} | {zs['total_tokens']:,} | "
            f"{zs['docs_with_tokens']:,} | {zs['avg_tokens_per_doc']:,.1f} | "
            f"{zs['vocabulary_size']:,} |"
        )
    lines.append("")
    lines.append("## Per-Category Statistics\n")
    lines.append("| Category | Docs | Total Tokens | Avg Tokens/Doc |")
    lines.append("|----------|------|-------------|----------------|")
    for cat, cs in stats["category_statistics"].items():
        lines.append(
            f"| {cat} | {cs['doc_count']} | {cs['total_tokens']:,} | "
            f"{cs['avg_tokens_per_doc']:,.1f} |"
        )
    return "\n".join(lines) + "\n"


def _generate_charts(
    stats: dict[str, Any],
    tokens_per_doc: list[int],
    charts_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate Phase 2 visualization charts."""
    try:
        # Chart 1: Token distribution per zone (bar chart)
        zone_stats = stats["zone_statistics"]
        zones = list(zone_stats.keys())
        zone_totals = [zone_stats[z]["total_tokens"] for z in zones]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            [z.capitalize() for z in zones], zone_totals,
            color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"],
        )
        ax.set_title("Total Tokens per Zone", fontsize=14, fontweight="bold")
        ax.set_ylabel("Token Count")
        for bar, val in zip(bars, zone_totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(charts_dir / "phase2_tokens_per_zone.png", dpi=150)
        plt.close()
        logger.info("Chart saved: phase2_tokens_per_zone.png")

        # Chart 2: Vocabulary size per zone
        vocab_sizes = [zone_stats[z]["vocabulary_size"] for z in zones]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            [z.capitalize() for z in zones], vocab_sizes,
            color=["#00BCD4", "#8BC34A", "#FFC107", "#E91E63", "#673AB7"],
        )
        ax.set_title("Vocabulary Size per Zone", fontsize=14, fontweight="bold")
        ax.set_ylabel("Unique Terms")
        for bar, val in zip(bars, vocab_sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(charts_dir / "phase2_vocab_per_zone.png", dpi=150)
        plt.close()
        logger.info("Chart saved: phase2_vocab_per_zone.png")

        # Chart 3: Tokens per document distribution (histogram)
        fig, ax = plt.subplots(figsize=(10, 6))
        # Cap at 99th percentile for visualization
        if tokens_per_doc:
            cap = sorted(tokens_per_doc)[int(len(tokens_per_doc) * 0.99)]
            capped = [min(t, cap) for t in tokens_per_doc]
            ax.hist(capped, bins=50, color="#3F51B5", edgecolor="white", alpha=0.85)
        ax.set_title("Document Token Count Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Tokens per Document")
        ax.set_ylabel("Number of Documents")
        ax.axvline(
            stats["average_tokens_per_doc"], color="red", linestyle="--",
            label=f"Mean: {stats['average_tokens_per_doc']:,.0f}",
        )
        ax.axvline(
            stats["median_tokens_per_doc"], color="green", linestyle="--",
            label=f"Median: {stats['median_tokens_per_doc']:,}",
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / "phase2_token_distribution.png", dpi=150)
        plt.close()
        logger.info("Chart saved: phase2_token_distribution.png")

        # Chart 4: Average tokens per category
        cat_stats = stats["category_statistics"]
        categories = list(cat_stats.keys())
        cat_avgs = [cat_stats[c]["avg_tokens_per_doc"] for c in categories]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(categories, cat_avgs, color="#FF5722")
        ax.set_title("Average Tokens per Document by Category", fontsize=14, fontweight="bold")
        ax.set_xlabel("Average Tokens/Doc")
        for bar, val in zip(bars, cat_avgs):
            ax.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {val:,.0f}", va="center", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(charts_dir / "phase2_tokens_by_category.png", dpi=150)
        plt.close()
        logger.info("Chart saved: phase2_tokens_by_category.png")

    except Exception as e:
        logger.error(f"Error generating Phase 2 charts: {e}")
