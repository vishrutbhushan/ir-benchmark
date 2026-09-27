from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Modern sleek styling defaults
PALETTE = ["#2563eb", "#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
BG_COLOR = "#ffffff"
TEXT_COLOR = "#1f2937"
GRID_COLOR = "#e5e7eb"


def _apply_plot_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="medium", color=TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="medium", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")


def generate_phase0_charts(audit_summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Generates visual charts for Phase 0 audit and returns paths to chart images."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Exclusion Categories Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    cats = audit_summary.get("exclusion_category_counts", {})
    clean_labels = {
        "empty_or_short_body": "Short (<200 char)",
        "js_stub_detected": "JS Render Stub",
        "non_html": "Non-HTML",
        "error_page": "404 / Error",
        "non_english": "Non-English",
        "normal_valid": "Clean / Valid",
    }
    labels = [clean_labels.get(k, k) for k in cats.keys()]
    counts = list(cats.values())
    colors = ["#f59e0b", "#ef4444", "#dc2626", "#b91c1c", "#8b5cf6", "#10b981"]

    bars = ax.barh(labels, counts, color=colors, edgecolor="none", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(counts) * 0.01 + 1, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=TEXT_COLOR)

    _apply_plot_style(ax, "Phase 0 Corpus Audit: Document Health & Exclusions", xlabel="Document Count")
    ax.set_xlim(0, max(counts) * 1.15 if max(counts) > 0 else 10)
    plt.tight_layout()
    chart_path = charts_dir / "phase0_exclusion_breakdown.png"
    plt.savefig(chart_path)
    plt.close(fig)
    generated["exclusion_breakdown"] = chart_path

    # 2. Top Technologies Average Character Lengths
    tech_breakdown = audit_summary.get("technology_breakdown", {})
    if tech_breakdown:
        top_techs = sorted(tech_breakdown.items(), key=lambda x: x[1].get("avg_char_length", 0), reverse=True)[:12]
        t_names = [t[0] for t in reversed(top_techs)]
        t_lengths = [t[1].get("avg_char_length", 0) for t in reversed(top_techs)]

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        bars = ax.barh(t_names, t_lengths, color="#3b82f6", edgecolor="none", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(t_lengths) * 0.01 + 10, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                    va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)

        _apply_plot_style(ax, "Average Document Length by Technology (Sampled)", xlabel="Average Characters")
        ax.set_xlim(0, max(t_lengths) * 1.18 if max(t_lengths) > 0 else 100)
        plt.tight_layout()
        chart_path = charts_dir / "phase0_tech_lengths.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["tech_lengths"] = chart_path

    return generated


def generate_phase1_charts(corpus_stats: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Generates visual charts for Phase 1 structured extraction and returns paths to chart images."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Category Distribution
    cat_dist = corpus_stats.get("category_distribution", {})
    if cat_dist:
        sorted_cats = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)
        cat_names = [c[0] for c in reversed(sorted_cats)]
        cat_counts = [c[1] for c in reversed(sorted_cats)]

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        bars = ax.barh(cat_names, cat_counts, color="#2563eb", edgecolor="none", height=0.65)
        for bar in bars:
            w = bar.get_width()
            pct = (w / sum(cat_counts)) * 100 if sum(cat_counts) else 0
            ax.text(w + max(cat_counts) * 0.01 + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({pct:.1f}%)",
                    va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)

        _apply_plot_style(ax, "Phase 1: Corpus Distribution Across Technical Categories", xlabel="Structured Documents")
        ax.set_xlim(0, max(cat_counts) * 1.25 if max(cat_counts) > 0 else 10)
        plt.tight_layout()
        chart_path = charts_dir / "phase1_category_distribution.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["category_distribution"] = chart_path

    # 2. Code Block Prevalence Donut Chart
    code_stats = corpus_stats.get("code_statistics", {})
    total_docs = corpus_stats.get("total_documents", 1)
    docs_with_code = code_stats.get("docs_with_code", 0)
    docs_without_code = max(0, total_docs - docs_with_code)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    sizes = [docs_with_code, docs_without_code]
    labels = [f"Contains Code ({docs_with_code:,})", f"Text Only ({docs_without_code:,})"]
    colors = ["#10b981", "#9ca3af"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
        textprops=dict(color=TEXT_COLOR, fontsize=9.5),
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Prevalence of Isolated Code Snippets in Documents", fontsize=12, fontweight="bold", pad=15, color=TEXT_COLOR)
    plt.tight_layout()
    chart_path = charts_dir / "phase1_code_prevalence.png"
    plt.savefig(chart_path)
    plt.close(fig)
    generated["code_prevalence"] = chart_path

    # 3. Top 15 Technologies by Volume
    tech_breakdown = corpus_stats.get("technology_breakdown", {})
    if tech_breakdown:
        top_techs = sorted(tech_breakdown.items(), key=lambda x: x[1].get("doc_count", 0), reverse=True)[:15]
        t_names = [t[0] for t in reversed(top_techs)]
        t_docs = [t[1].get("doc_count", 0) for t in reversed(top_techs)]

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        bars = ax.barh(t_names, t_docs, color="#06b6d4", edgecolor="none", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(t_docs) * 0.01 + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(w)} docs",
                    va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)

        _apply_plot_style(ax, "Phase 1: Documents Extracted by Technology", xlabel="Extracted Documents")
        ax.set_xlim(0, max(t_docs) * 1.25 if max(t_docs) > 0 else 10)
        plt.tight_layout()
        chart_path = charts_dir / "phase1_top_technologies.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["top_technologies"] = chart_path

    return generated


def generate_interactive_dashboard_html(
    phase0_summary: dict[str, Any] | None,
    phase1_summary: dict[str, Any] | None,
    output_html_path: Path,
) -> None:
    """Creates a standalone interactive HTML visual dashboard with Tailwind CSS."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    p0_json = json.dumps(phase0_summary or {}, ensure_ascii=False)
    p1_json = json.dumps(phase1_summary or {}, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IR Benchmark Analytical Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-slate-50 text-slate-900 antialiased p-6 min-h-screen">
  <div class="max-w-7xl mx-auto space-y-6">
    
    <!-- Header -->
    <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-slate-900">Technical IR Benchmark Dashboard</h1>
        <p class="text-slate-500 text-sm mt-1">Real-time analytical visualization of corpus audit, extraction, and content metrics.</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
          ● Phase 0 & 1 Ready
        </span>
      </div>
    </div>

    <!-- KPI Summary Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Sample Analyzed</span>
        <div class="mt-2 text-2xl font-extrabold text-blue-600" id="kpi-sample">--</div>
        <p class="text-xs text-slate-400 mt-1">Documents audited</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Clean Documents</span>
        <div class="mt-2 text-2xl font-extrabold text-emerald-600" id="kpi-clean">--</div>
        <p class="text-xs text-slate-400 mt-1">Surviving quality checks</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Code Prevalence</span>
        <div class="mt-2 text-2xl font-extrabold text-purple-600" id="kpi-code">--</div>
        <p class="text-xs text-slate-400 mt-1">Docs with code blocks</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Near-Duplicate Clusters</span>
        <div class="mt-2 text-2xl font-extrabold text-amber-600" id="kpi-clusters">--</div>
        <p class="text-xs text-slate-400 mt-1">SimHash Hamming ≤ 3</p>
      </div>
    </div>

    <!-- Visual Charts Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      
      <!-- Phase 0 Exclusions Chart Card -->
      <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h2 class="text-base font-bold text-slate-900 mb-1">Phase 0: Exclusion Category Breakdown</h2>
        <p class="text-xs text-slate-500 mb-4">Distribution of document quality flags across sample.</p>
        <div class="flex items-center justify-center p-2">
          <img src="phase0/charts/phase0_exclusion_breakdown.png" alt="Exclusion Breakdown" class="rounded-lg max-h-72 object-contain" onerror="this.parentElement.innerHTML='<span class=\\'text-xs text-slate-400\\'>Chart not yet generated.</span>'"/>
        </div>
      </div>

      <!-- Phase 1 Categories Chart Card -->
      <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h2 class="text-base font-bold text-slate-900 mb-1">Phase 1: Category Distribution</h2>
        <p class="text-xs text-slate-500 mb-4">Structured document count by technical domain.</p>
        <div class="flex items-center justify-center p-2">
          <img src="phase1/charts/phase1_category_distribution.png" alt="Category Distribution" class="rounded-lg max-h-72 object-contain" onerror="this.parentElement.innerHTML='<span class=\\'text-xs text-slate-400\\'>Chart not yet generated.</span>'"/>
        </div>
      </div>

      <!-- Phase 1 Code Prevalence Card -->
      <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h2 class="text-base font-bold text-slate-900 mb-1">Phase 1: Code Block Extraction</h2>
        <p class="text-xs text-slate-500 mb-4">Share of documents with isolated technical code snippets.</p>
        <div class="flex items-center justify-center p-2">
          <img src="phase1/charts/phase1_code_prevalence.png" alt="Code Prevalence" class="rounded-lg max-h-72 object-contain" onerror="this.parentElement.innerHTML='<span class=\\'text-xs text-slate-400\\'>Chart not yet generated.</span>'"/>
        </div>
      </div>

      <!-- Phase 0 Document Lengths Card -->
      <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h2 class="text-base font-bold text-slate-900 mb-1">Phase 0: Document Lengths by Technology</h2>
        <p class="text-xs text-slate-500 mb-4">Average character length across top sampled technologies.</p>
        <div class="flex items-center justify-center p-2">
          <img src="phase0/charts/phase0_tech_lengths.png" alt="Tech Document Lengths" class="rounded-lg max-h-72 object-contain" onerror="this.parentElement.innerHTML='<span class=\\'text-xs text-slate-400\\'>Chart not yet generated.</span>'"/>
        </div>
      </div>

    </div>

    <!-- Recommendations / Decision Panel -->
    <div class="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
      <h3 class="text-base font-bold text-slate-900 mb-2">Human-in-the-Loop Sign-off Decisions</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-3">
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-100">
          <span class="font-semibold text-slate-800">1. Body Length Threshold</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Exclude &lt; 200 characters (0.8% of documents flagged as empty/stubs).</p>
        </div>
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-100">
          <span class="font-semibold text-slate-800">2. Language Filter</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Filter non-English pages (5.2% detected, e.g., Japanese Oracle docs).</p>
        </div>
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-100">
          <span class="font-semibold text-slate-800">3. Deduplication</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Collapse SimHash clusters (Hamming ≤ 3) to 1 representative document.</p>
        </div>
      </div>
    </div>

  </div>

  <script>
    const p0 = {p0_json};
    const p1 = {p1_json};

    if (p0 && p0.sample_size) {{
      document.getElementById('kpi-sample').innerText = Number(p0.sample_size).toLocaleString();
      const valid = p0.exclusion_category_counts?.normal_valid || 0;
      const validPct = p0.exclusion_percentages?.normal_valid || 0;
      document.getElementById('kpi-clean').innerText = `${{Number(valid).toLocaleString()}} (${{validPct}}%)`;
      document.getElementById('kpi-clusters').innerText = Number(p0.near_duplicate_stats?.clusters_count || 0).toLocaleString();
    }}

    if (p1 && p1.code_statistics) {{
      const pct = p1.code_statistics.pct_docs_with_code || 0;
      document.getElementById('kpi-code').innerText = `${{pct}}%`;
    }}
  </script>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated interactive dashboard HTML at {output_html_path}")
