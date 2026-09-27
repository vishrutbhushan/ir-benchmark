from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from src.utils.file_utils import stream_jsonl, write_json, write_text
from src.utils.visualizer import generate_phase1_charts


def compute_corpus_statistics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculates comprehensive statistics over extracted structured records."""
    total_docs = 0
    total_chars = 0
    total_words = 0
    total_code_chars = 0
    total_code_blocks = 0
    docs_with_code = 0
    docs_with_dates = 0
    total_dates_detected = 0

    tech_stats: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"docs": 0, "chars": 0, "words": 0, "code_blocks": 0, "dates": 0}
    )
    category_stats: dict[str, int] = collections.defaultdict(int)
    date_counts: collections.Counter = collections.Counter()

    for r in records:
        total_docs += 1
        meta = r.get("metadata", {})
        tech = meta.get("technology", "Unknown")
        cat = meta.get("category", "Unknown")

        body = r.get("body_text", "")
        code = r.get("code_text", "")
        dates = r.get("dates", [])
        code_blocks_count = meta.get("code_blocks_count", 0)

        chars = len(body)
        words = len(body.split())
        code_chars = len(code)

        total_chars += chars
        total_words += words
        total_code_chars += code_chars
        total_code_blocks += code_blocks_count

        if code_blocks_count > 0 or code_chars > 0:
            docs_with_code += 1

        if dates:
            docs_with_dates += 1
            total_dates_detected += len(dates)
            date_counts.update(dates)

        # Tech breakdown
        tech_stats[tech]["docs"] += 1
        tech_stats[tech]["chars"] += chars
        tech_stats[tech]["words"] += words
        tech_stats[tech]["code_blocks"] += code_blocks_count
        tech_stats[tech]["dates"] += len(dates)

        category_stats[cat] += 1

    summary = {
        "total_documents": total_docs,
        "aggregate_body_chars": total_chars,
        "aggregate_body_words": total_words,
        "average_doc_chars": round(total_chars / total_docs, 1) if total_docs else 0,
        "average_doc_words": round(total_words / total_docs, 1) if total_docs else 0,
        "code_statistics": {
            "docs_with_code": docs_with_code,
            "pct_docs_with_code": round((docs_with_code / total_docs) * 100, 2) if total_docs else 0,
            "total_code_blocks": total_code_blocks,
            "total_code_characters": total_code_chars,
            "avg_code_blocks_per_doc": round(total_code_blocks / total_docs, 2) if total_docs else 0,
        },
        "date_statistics": {
            "docs_with_dates": docs_with_dates,
            "pct_docs_with_dates": round((docs_with_dates / total_docs) * 100, 2) if total_docs else 0,
            "total_dates_detected": total_dates_detected,
            "most_common_dates": date_counts.most_common(10),
        },
        "category_distribution": dict(category_stats),
        "technology_breakdown": {
            t: {
                "doc_count": s["docs"],
                "avg_chars": round(s["chars"] / s["docs"], 1) if s["docs"] else 0,
                "avg_words": round(s["words"] / s["docs"], 1) if s["docs"] else 0,
                "total_code_blocks": s["code_blocks"],
                "total_dates": s["dates"],
            }
            for t, s in sorted(tech_stats.items(), key=lambda x: x[1]["docs"], reverse=True)
        },
    }
    return summary


def generate_markdown_statistics_report(stats: dict[str, Any]) -> str:
    """Formats corpus statistics into a clean markdown analytical report."""
    total_docs = stats["total_documents"]
    code = stats["code_statistics"]
    dates = stats["date_statistics"]

    lines = [
        "# Phase 1 — Structured Corpus Statistics Report",
        "",
        f"**Total Structured Documents Processed:** {total_docs:,}",
        f"**Average Document Length:** {stats['average_doc_words']:,} words ({stats['average_doc_chars']:,} characters)",
        f"**Total Vocabulary / Body Words:** {stats['aggregate_body_words']:,}",
        "",
        "## Visual Charts & Graphs",
        "",
        "![Phase 1 Category Distribution](charts/phase1_category_distribution.png)",
        "",
        "![Phase 1 Code Prevalence](charts/phase1_code_prevalence.png)",
        "",
        "![Phase 1 Top Technologies](charts/phase1_top_technologies.png)",
        "",
        "## 1. Code Block & Technical Token Extraction",
        "",
        f"- **Documents with Code Blocks:** {code['docs_with_code']:,} ({code['pct_docs_with_code']}%)",
        f"- **Total Isolated Code Blocks:** {code['total_code_blocks']:,}",
        f"- **Average Code Blocks per Document:** {code['avg_code_blocks_per_doc']}",
        f"- **Total Code Snippet Characters:** {code['total_code_characters']:,}",
        "",
        "## 2. Date Detection & Normalization",
        "",
        f"- **Documents with Dates Detected:** {dates['docs_with_dates']:,} ({dates['pct_docs_with_dates']}%)",
        f"- **Total Normalized Dates Extracted:** {dates['total_dates_detected']:,}",
        "",
        "### Top Normalized Dates Detected",
        "",
        "| Date (ISO) | Document Frequency |",
        "| :--- | :--- |",
    ]

    for d, freq in dates.get("most_common_dates", [])[:8]:
        lines.append(f"| `{d}` | {freq:,} |")

    lines.extend([
        "",
        "## 3. Category Breakdown",
        "",
        "| Category | Documents | Share |",
        "| :--- | :--- | :--- |",
    ])

    for cat, count in stats["category_distribution"].items():
        share = round((count / total_docs) * 100, 1) if total_docs else 0
        lines.append(f"| **{cat}** | {count:,} | {share}% |")

    lines.extend([
        "",
        "## 4. Technology Breakdown (Top 20 by Volume)",
        "",
        "| Technology | Docs | Avg Words | Avg Chars | Code Blocks | Dates |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for tech, tstats in list(stats["technology_breakdown"].items())[:20]:
        lines.append(
            f"| {tech} | {tstats['doc_count']:,} | {tstats['avg_words']} | {tstats['avg_chars']} | {tstats['total_code_blocks']:,} | {tstats['total_dates']} |"
        )

    return "\n".join(lines)


def generate_and_save_statistics(records_path: Path, output_dir: Path) -> dict[str, Any]:
    """Reads structured records file and produces JSON, Markdown, and Visual Chart reports."""
    stats = compute_corpus_statistics(stream_jsonl(records_path))

    # Generate charts
    try:
        generate_phase1_charts(stats, output_dir)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to generate Phase 1 charts: {e}")

    write_json(stats, output_dir / "corpus_statistics.json")
    md_content = generate_markdown_statistics_report(stats)
    write_text(md_content, output_dir / "corpus_statistics.md")
    return stats
