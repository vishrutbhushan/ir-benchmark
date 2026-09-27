from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

from src.phase0_audit.content_validator import detect_error_page, detect_language, is_non_html
from src.phase0_audit.duplicate_detector import find_near_duplicate_clusters
from src.phase0_audit.sampler import extract_body_text_metrics, sample_manifest_documents
from src.utils.file_utils import write_json, write_jsonl, write_text
from src.utils.visualizer import generate_phase0_charts


def run_corpus_audit(
    corpus_root: Path,
    manifest_path: Path,
    output_dir: Path,
    logger: logging.Logger,
    sample_size_per_tech: int = 15,
    total_sample_limit: int = 1000,
    min_char_length_threshold: int = 200,
) -> dict[str, Any]:
    """Runs Phase 0 corpus audit and generates analytical outputs for human sign-off."""
    logger.info("Starting Phase 0 Corpus Audit...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified sampling
    logger.info(
        f"Sampling manifest ({manifest_path}) with up to {sample_size_per_tech} docs/tech, max {total_sample_limit} total..."
    )
    sampled_manifest = sample_manifest_documents(
        manifest_path=manifest_path,
        sample_size_per_tech=sample_size_per_tech,
        total_sample_limit=total_sample_limit,
    )
    logger.info(f"Sampled {len(sampled_manifest)} documents across technologies.")

    # 2. Inspect each document
    audit_records: list[dict[str, Any]] = []
    category_counts = {
        "empty_or_short_body": 0,
        "js_stub_detected": 0,
        "non_html": 0,
        "error_page": 0,
        "non_english": 0,
        "normal_valid": 0,
    }
    flagged_samples: list[dict[str, Any]] = []
    tech_stats: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"total": 0, "short": 0, "non_eng": 0, "error": 0, "lengths": []}
    )

    docs_for_dedup: list[dict[str, Any]] = []

    for idx, record in enumerate(sampled_manifest, start=1):
        doc_id = record["doc_id"]
        rel_html_path = record["html_path"]
        tech = record.get("technology", "Unknown")
        full_html_path = corpus_root / rel_html_path

        tech_stats[tech]["total"] += 1

        if not full_html_path.exists():
            logger.warning(f"File missing on disk: {full_html_path}")
            continue

        try:
            html_content = full_html_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error reading {full_html_path}: {e}")
            continue

        # Non-HTML check
        non_html_flag = is_non_html(html_content, record.get("content_mime_type"))
        if non_html_flag:
            category_counts["non_html"] += 1

        # Body text & length metrics
        metrics = extract_body_text_metrics(html_content)
        body_text = metrics["body_text"]
        char_len = metrics["char_length"]
        tech_stats[tech]["lengths"].append(char_len)

        is_short = char_len < min_char_length_threshold
        if is_short:
            category_counts["empty_or_short_body"] += 1
            tech_stats[tech]["short"] += 1

        if metrics["js_stub_detected"]:
            category_counts["js_stub_detected"] += 1

        # Error / 404 detection
        is_error, error_reason = detect_error_page(record.get("title", ""), body_text)
        if is_error:
            category_counts["error_page"] += 1
            tech_stats[tech]["error"] += 1

        # Language detection
        detected_lang, is_eng = detect_language(body_text, record.get("content_languages"))
        if not is_eng:
            category_counts["non_english"] += 1
            tech_stats[tech]["non_eng"] += 1

        # Audit verdict
        is_flagged = is_short or metrics["js_stub_detected"] or non_html_flag or is_error or not is_eng
        if not is_flagged:
            category_counts["normal_valid"] += 1

        doc_summary = {
            "doc_id": doc_id,
            "technology": tech,
            "title": record.get("title", ""),
            "url": record.get("url", ""),
            "char_length": char_len,
            "word_count": metrics["word_count"],
            "is_short": is_short,
            "js_stub_detected": metrics["js_stub_detected"],
            "non_html": non_html_flag,
            "is_error": is_error,
            "error_reason": error_reason,
            "detected_lang": detected_lang,
            "is_english": is_eng,
            "html_path": str(rel_html_path),
        }
        audit_records.append(doc_summary)

        if is_flagged and len(flagged_samples) < 100:
            flagged_samples.append({**doc_summary, "body_preview": body_text[:250]})

        if not is_short:
            docs_for_dedup.append({
                "doc_id": doc_id,
                "technology": tech,
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "body_text": body_text,
            })

    # 3. Near-duplicate clustering
    logger.info("Running near-duplicate clustering using SimHash...")
    dedup_results = find_near_duplicate_clusters(docs_for_dedup, max_hamming_distance=3)
    logger.info(
        f"Found {dedup_results['total_clusters']} duplicate clusters ({dedup_results['total_duplicate_documents']} docs involved)."
    )

    # 4. Generate structured report data
    total_docs = len(audit_records)
    audit_summary = {
        "sample_size": total_docs,
        "min_char_length_threshold": min_char_length_threshold,
        "exclusion_category_counts": category_counts,
        "exclusion_percentages": {
            k: round((v / total_docs) * 100, 2) if total_docs else 0
            for k, v in category_counts.items()
        },
        "near_duplicate_stats": {
            "analyzed_docs": dedup_results["total_documents_analyzed"],
            "clusters_count": dedup_results["total_clusters"],
            "duplicate_docs_count": dedup_results["total_duplicate_documents"],
        },
        "technology_breakdown": {
            t: {
                "sample_count": s["total"],
                "short_count": s["short"],
                "non_english_count": s["non_eng"],
                "error_count": s["error"],
                "avg_char_length": round(sum(s["lengths"]) / len(s["lengths"]), 1) if s["lengths"] else 0,
            }
            for t, s in tech_stats.items()
        },
        "example_flagged_docs": {
            "empty_or_short_body": [d["doc_id"] for d in audit_records if d["is_short"]][:5],
            "js_stub_detected": [d["doc_id"] for d in audit_records if d["js_stub_detected"]][:5],
            "non_html": [d["doc_id"] for d in audit_records if d["non_html"]][:5],
            "error_page": [d["doc_id"] for d in audit_records if d["is_error"]][:5],
            "non_english": [d["doc_id"] for d in audit_records if not d["is_english"]][:5],
        },
    }

    # Generate visual charts
    try:
        charts = generate_phase0_charts(audit_summary, output_dir)
        logger.info(f"Generated {len(charts)} visual charts in {output_dir / 'charts'}")
    except Exception as e:
        logger.warning(f"Could not generate visual charts: {e}")

    # Write JSON outputs
    write_json(audit_summary, output_dir / "audit_report.json")
    write_json(dedup_results, output_dir / "duplicate_clusters.json")
    write_jsonl(flagged_samples, output_dir / "sample_inspections.jsonl")

    # Generate Markdown report
    md_report = generate_markdown_audit_report(audit_summary, dedup_results)
    write_text(md_report, output_dir / "audit_report.md")

    logger.info(f"Phase 0 audit complete. Reports saved to {output_dir}")
    return audit_summary


def generate_markdown_audit_report(summary: dict[str, Any], dedup: dict[str, Any]) -> str:
    """Formats the Phase 0 audit results into an analytical markdown report."""
    cats = summary["exclusion_category_counts"]
    pcts = summary["exclusion_percentages"]
    sample_size = summary["sample_size"]

    lines = [
        "# Phase 0 — Corpus Audit Report",
        "",
        f"**Sample Size Analyzed:** {sample_size} documents across technologies.",
        f"**Text-Length Threshold:** < {summary['min_char_length_threshold']} characters.",
        "",
        "## Visual Charts & Graphs",
        "",
        "![Phase 0 Exclusions Breakdown](charts/phase0_exclusion_breakdown.png)",
        "",
        "![Phase 0 Document Lengths](charts/phase0_tech_lengths.png)",
        "",
        "## 1. Exclusion Categories Summary",
        "",
        "| Category | Flagged Count | Sample % | Example Doc IDs |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Empty / Short Body (<{summary['min_char_length_threshold']} chars)** | {cats['empty_or_short_body']} | {pcts['empty_or_short_body']}% | {', '.join(summary['example_flagged_docs']['empty_or_short_body']) or 'None'} |",
        f"| **Client-Side JS Render Stubs** | {cats['js_stub_detected']} | {pcts['js_stub_detected']}% | {', '.join(summary['example_flagged_docs']['js_stub_detected']) or 'None'} |",
        f"| **Non-HTML Content Mislabeled** | {cats['non_html']} | {pcts['non_html']}% | {', '.join(summary['example_flagged_docs']['non_html']) or 'None'} |",
        f"| **HTTP 200 Error / 404 Pages** | {cats['error_page']} | {pcts['error_page']}% | {', '.join(summary['example_flagged_docs']['error_page']) or 'None'} |",
        f"| **Non-English Documents** | {cats['non_english']} | {pcts['non_english']}% | {', '.join(summary['example_flagged_docs']['non_english']) or 'None'} |",
        f"| **Clean / Valid Documents** | {cats['normal_valid']} | {pcts['normal_valid']}% | — |",
        "",
        "## 2. Near-Duplicate Analysis (SimHash Clustering)",
        "",
        f"- **Documents Analyzed for Near-Duplicates:** {dedup['total_documents_analyzed']}",
        f"- **Near-Duplicate Clusters Found:** {dedup['total_clusters']}",
        f"- **Total Documents in Duplicate Clusters:** {dedup['total_duplicate_documents']}",
        "",
        "### Top Near-Duplicate Cluster Samples",
        "",
    ]

    for cluster in dedup.get("clusters", [])[:5]:
        lines.append(
            f"* **Cluster {cluster['cluster_id']}** ({cluster['cluster_size']} docs) — Representative: `{cluster['representative_doc_id']}` (`{cluster['representative_title']}`)"
        )
        lines.append(f"  - Members: {', '.join(cluster['member_doc_ids'][:6])}")

    lines.extend([
        "",
        "## 3. Technology Distribution & Character Lengths (Top 15 Technologies in Sample)",
        "",
        "| Technology | Sample Docs | Avg Char Length | Short (<200) | Non-English | Errors |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    top_techs = sorted(
        summary["technology_breakdown"].items(),
        key=lambda x: x[1]["sample_count"],
        reverse=True,
    )[:15]

    for tech, stats in top_techs:
        lines.append(
            f"| {tech} | {stats['sample_count']} | {stats['avg_char_length']} | {stats['short_count']} | {stats['non_english_count']} | {stats['error_count']} |"
        )

    lines.extend([
        "",
        "## 4. Key Recommendations for Human Sign-off",
        "",
        "1. **Empty/JS Stub Exclusion:** Exclude all documents with body text < 200 characters or explicit JS stub indicators (`<div id='root'></div>` with no extracted text).",
        "2. **Error / 404 Filter:** Drop documents matching error/redirect page signatures.",
        "3. **Language Filter:** Require English text (`en` detected, or manifest `eng` when text is too short).",
        "4. **Duplicate Deduplication:** Retain only the first/representative document from each near-duplicate SimHash cluster.",
        "5. **Spot-Check File:** Review `outputs/phase0/sample_inspections.jsonl` to verify flagged samples.",
        "",
    ])

    return "\n".join(lines)
