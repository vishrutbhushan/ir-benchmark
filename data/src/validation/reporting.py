from __future__ import annotations

import html
import json
from pathlib import Path


def write_report(summary: dict, examples: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = output_dir / "charts"
    charts.mkdir(exist_ok=True)
    chart_specs = [
        (
            "document_status",
            "Every document: integrity and content triage",
            summary["status_counts"],
        ),
        ("category_distribution", "Full manifest categories", summary["categories"]),
        (
            "quality_reasons",
            "Exclusion/review reasons (may overlap)",
            summary["quality_reasons"],
        ),
        (
            "language_evidence",
            "Language evidence across every document",
            summary["languages"],
        ),
        (
            "code_prevalence",
            "HTML documents with and without code",
            summary["code_prevalence"],
        ),
    ]
    for name, title, counts in chart_specs:
        items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:25]
        fig, ax = plt.subplots(figsize=(10, max(3, len(items) * 0.33)))
        if items:
            ax.barh(
                [k for k, _ in items][::-1],
                [v for _, v in items][::-1],
                color="#3178ad",
            )
        ax.set_title(title)
        ax.set_xlabel("Documents")
        fig.tight_layout()
        fig.savefig(charts / f"{name}.png", dpi=140)
        plt.close(fig)

    statuses = ["eligible", "review", "exclude", "failed"]
    categories = list(summary["category_status"])
    fig, ax = plt.subplots(figsize=(10, 5))
    left = [0] * len(categories)
    for status in statuses:
        values = [
            summary["category_status"][category].get(status, 0)
            for category in categories
        ]
        ax.barh(categories, values, left=left, label=status)
        left = [a + b for a, b in zip(left, values)]
    ax.set_title("Content eligibility by category (all documents)")
    ax.set_xlabel("Documents")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "category_eligibility.png", dpi=140)
    plt.close(fig)

    inventory = summary["inventory"]
    text = (
        "# Full raw-corpus validation\n\n"
        f"Completed: **{summary['completed']}**. Examined **{summary['processed_documents']:,} / "
        f"{inventory['manifest_documents']:,} documents**. No sampling.\n\n"
        f"Integrity gate: **{summary['integrity_gate_passed']}**. "
        f"CRC/size/SHA or materialization failures: **{summary['integrity_failures']}**.\n\n"
        f"Status counts: `{json.dumps(summary['status_counts'])}`. "
        "Content eligibility is an automated policy, not a guarantee of semantic correctness.\n\n"
        "Every HTML member was read completely, checked against ZIP CRC, manifest byte length and SHA-256, "
        "parsed, inspected and safely materialized with normalized POSIX paths. Other ZIP files were fully "
        "read for CRC verification. Raw HTML bytes and the ZIP were not rewritten.\n\n"
        "`integrity_manifest.jsonl` contains all integrity-valid documents; `eligible_manifest.jsonl` "
        "contains provisional content-eligible documents. `review_manifest.jsonl` holds uncertain cases; "
        "`excluded_manifest.jsonl` holds reasoned exclusions. Do not freeze an evaluation corpus until "
        "review/policy decisions are resolved. `document_checks.jsonl` records every decision.\n\n"
        "HTML recovery/omitted closing tags do not by themselves establish corruption. UTF-8 failures can "
        "indicate valid legacy encodings. Language evidence trusts agreement of HTML and manifest English "
        "labels; other substantial prose is checked with a deterministic detector. A page with matching "
        "download-time hashes can still be a bad source capture. Only comparison with the original WARC "
        "can establish capture fidelity beyond the supplied archive/manifest.\n\n"
        "Text-identical groups are review candidates, not automatically deleted: different version URLs "
        "can contain identical text. This pass does not compute a neural representation or perform "
        "full near-duplicate semantic detection.\n\n"
    )
    text += (
        f"Exact visible-text candidate groups: **{summary['identical_visible_content_groups']:,}**, "
        f"covering {summary['documents_in_identical_visible_content_groups']:,} documents; not automatically removed.\n\n"
        f"Technologies with no eligible page: `{summary['technologies_without_eligible_documents']}`. "
        "Kafka's three captures are redirect stubs, not usable documentation bodies.\n\n"
        "The default normalized raw manifest retains all 79,268 files. Both engines must use the same "
        "explicitly frozen eligibility manifest later; review pages are not silently admitted.\n\n"
        "![Eligibility by category](charts/category_eligibility.png)\n\n"
    )
    for name, title, _ in chart_specs:
        text += f"![{title}](charts/{name}.png)\n\n"
    text += "[Interactive report](dashboard.html) · [Machine-readable summary](summary.json)\n"
    (output_dir / "report.md").write_text(text, encoding="utf-8")
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key in ["doc_id", "technology", "status", "reasons", "flags"]
        )
        + "</tr>"
        for row in examples
    )
    (output_dir / "dashboard.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Full corpus validation</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
        "img{max-width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}</style>"
        "<h1>Full raw-corpus validation</h1>"
        f"<p>Processed {summary['processed_documents']:,} documents; no sampling. "
        f"Integrity gate passed: {summary['integrity_gate_passed']}.</p>"
        "<p>All decisions: <a href='document_checks.jsonl'>document_checks.jsonl</a>. "
        "The table shows up to five examples per technology/status, not the inspection population.</p>"
        "<img alt='Eligibility by category' src='charts/category_eligibility.png'>"
        + "".join(
            f"<img alt='{html.escape(title)}' src='charts/{name}.png'>"
            for name, title, _ in chart_specs
        )
        + "<label>Filter examples <input id='filter'></label><table><thead><tr>"
        "<th>Document</th><th>Technology</th><th>Status</th><th>Reasons</th><th>Flags</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
        "<script>document.querySelector('#filter').addEventListener('input',e=>{"
        "for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase()"
        ".includes(e.target.value.toLowerCase())})</script>",
        encoding="utf-8",
    )
