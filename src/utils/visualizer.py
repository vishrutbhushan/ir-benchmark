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
    """Generates comprehensive visual charts for Phase 0 audit and returns paths to chart images."""
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

    # 2. Language Distribution (Donut Chart)
    lang_stats = audit_summary.get("language_distribution", {})
    if lang_stats:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        sorted_langs = sorted(lang_stats.items(), key=lambda x: x[1], reverse=True)
        top_langs = sorted_langs[:6]
        other_sum = sum(x[1] for x in sorted_langs[6:])
        if other_sum > 0:
            top_langs.append(("Other", other_sum))

        l_names = [x[0] for x in top_langs]
        l_counts = [x[1] for x in top_langs]
        l_colors = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#9ca3af"][:len(top_langs)]

        wedges, texts, autotexts = ax.pie(
            l_counts, labels=l_names, autopct="%1.1f%%", startangle=140, colors=l_colors,
            textprops=dict(color=TEXT_COLOR, fontsize=9),
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2)
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(8.5)
        ax.set_title("Phase 0: Language Distribution Across Sample", fontsize=12, fontweight="bold", pad=15, color=TEXT_COLOR)
        plt.tight_layout()
        chart_path = charts_dir / "phase0_language_distribution.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["language_distribution"] = chart_path

    # 3. Code vs No-Code Presence by Category (100% Stacked Bar)
    cat_code = audit_summary.get("category_code_breakdown", {})
    if cat_code:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        categories = list(cat_code.keys())
        has_code_pcts = [cat_code[c].get("pct_with_code", 0) for c in categories]
        no_code_pcts = [100 - p for p in has_code_pcts]

        y_pos = range(len(categories))
        ax.barh(y_pos, has_code_pcts, color="#10b981", label="Contains Code", height=0.6)
        ax.barh(y_pos, no_code_pcts, left=has_code_pcts, color="#9ca3af", label="Text Only", height=0.6)

        for i, (pct, total) in enumerate(zip(has_code_pcts, [cat_code[c].get("total", 0) for c in categories])):
            if pct > 15:
                ax.text(pct / 2, i, f"{pct:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=8.5)
            ax.text(102, i, f"n={total}", va="center", ha="left", color=TEXT_COLOR, fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlim(0, 115)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        _apply_plot_style(ax, "Phase 0: Code Block Prevalence Across Technical Categories (%)", xlabel="Percentage of Documents (%)")
        plt.tight_layout()
        chart_path = charts_dir / "phase0_code_by_category.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["code_by_category"] = chart_path

    # 4. Boilerplate Overhead Ratio by Category
    bp_breakdown = audit_summary.get("category_boilerplate_breakdown", {})
    if bp_breakdown:
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        sorted_bp = sorted(bp_breakdown.items(), key=lambda x: x[1].get("avg_ratio", 0), reverse=True)
        b_names = [b[0] for b in reversed(sorted_bp)]
        b_ratios = [round(b[1].get("avg_ratio", 0) * 100, 1) for b in reversed(sorted_bp)]

        bars = ax.barh(b_names, b_ratios, color="#f59e0b", edgecolor="none", height=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                    va="center", ha="left", fontsize=8.5, color=TEXT_COLOR, fontweight="bold")

        _apply_plot_style(ax, "HTML Boilerplate & Markup Overhead Ratio by Category", xlabel="Overhead Share (% of Raw HTML)")
        ax.set_xlim(0, 110)
        plt.tight_layout()
        chart_path = charts_dir / "phase0_boilerplate_by_category.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["boilerplate_by_category"] = chart_path

    # 5. DOM Structure Complexity (Headings, Code, Tables, Lists)
    dom_stats = audit_summary.get("dom_structure_stats", {})
    if dom_stats:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        features = ["Headings (H1-H3)", "Code Blocks", "Lists (UL/OL)", "Tables"]
        pcts = [
            dom_stats.get("pct_with_headings", 0),
            dom_stats.get("pct_with_code", 0),
            dom_stats.get("pct_with_lists", 0),
            dom_stats.get("pct_with_tables", 0),
        ]
        bars = ax.bar(features, pcts, color=["#2563eb", "#10b981", "#8b5cf6", "#06b6d4"], width=0.55)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_COLOR)

        _apply_plot_style(ax, "Phase 0: Document Structural Element Prevalence", ylabel="Share of Documents (%)")
        ax.set_ylim(0, 115)
        plt.tight_layout()
        chart_path = charts_dir / "phase0_dom_complexity.png"
        plt.savefig(chart_path)
        plt.close(fig)
        generated["dom_complexity"] = chart_path

    # 6. Top Technologies Average Character Lengths
    tech_breakdown = audit_summary.get("technology_breakdown", {})
    if tech_breakdown:
        top_techs = sorted(tech_breakdown.items(), key=lambda x: x[1].get("avg_char_length", 0), reverse=True)[:15]
        t_names = [t[0] for t in reversed(top_techs)]
        t_lengths = [t[1].get("avg_char_length", 0) for t in reversed(top_techs)]

        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
        bars = ax.barh(t_names, t_lengths, color="#3b82f6", edgecolor="none", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(t_lengths) * 0.01 + 10, bar.get_y() + bar.get_height() / 2, f"{int(w):,}",
                    va="center", ha="left", fontsize=8.5, color=TEXT_COLOR)

        _apply_plot_style(ax, "Phase 0: Average Document Length by Technology", xlabel="Average Characters")
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
        <p class="text-slate-500 text-sm mt-1">Multi-dimensional analytical visualizations for corpus audit, language distribution, code presence, and extraction metrics.</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
          ● Phase 0 & 1 Ready
        </span>
      </div>
    </div>

    <!-- KPI Summary Grid -->
    <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Sample Audited</span>
        <div class="mt-2 text-2xl font-extrabold text-blue-600" id="kpi-sample">--</div>
        <p class="text-xs text-slate-400 mt-1">Across 11 categories</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Clean Documents</span>
        <div class="mt-2 text-2xl font-extrabold text-emerald-600" id="kpi-clean">--</div>
        <p class="text-xs text-slate-400 mt-1">Passed all flags</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Code Prevalence</span>
        <div class="mt-2 text-2xl font-extrabold text-purple-600" id="kpi-code">--</div>
        <p class="text-xs text-slate-400 mt-1">Documents with code</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Non-English Rate</span>
        <div class="mt-2 text-2xl font-extrabold text-amber-600" id="kpi-lang">--</div>
        <p class="text-xs text-slate-400 mt-1">Multilingual mirrors</p>
      </div>
      <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
        <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Near-Duplicate Clusters</span>
        <div class="mt-2 text-2xl font-extrabold text-red-600" id="kpi-clusters">--</div>
        <p class="text-xs text-slate-400 mt-1">SimHash Hamming ≤ 3</p>
      </div>
    </div>

    <!-- Section 1: Phase 0 Visual Analytics -->
    <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
      <div class="border-b border-slate-100 pb-3">
        <h2 class="text-lg font-bold text-slate-900">Phase 0: Corpus Audit & Structural Analytics</h2>
        <p class="text-xs text-slate-500">Comprehensive breakdown of document quality, multilingual presence, code density, and DOM characteristics.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-2">
        <!-- 1. Exclusions -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">1. Health & Exclusions</h3>
          <p class="text-xs text-slate-500 mb-3">Document quality & defect flags.</p>
          <img src="phase0/charts/phase0_exclusion_breakdown.png" alt="Exclusion Breakdown" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <!-- 2. Language Distribution -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">2. Language Distribution</h3>
          <p class="text-xs text-slate-500 mb-3">Multilingual shares in the corpus.</p>
          <img src="phase0/charts/phase0_language_distribution.png" alt="Language Distribution" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <!-- 3. Code vs No-Code Presence -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">3. Code vs. Text by Category</h3>
          <p class="text-xs text-slate-500 mb-3">Prevalence of code blocks across domains.</p>
          <img src="phase0/charts/phase0_code_by_category.png" alt="Code Prevalence by Category" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <!-- 4. Boilerplate Overhead -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">4. HTML Boilerplate Overhead</h3>
          <p class="text-xs text-slate-500 mb-3">Markup & chrome percentage by category.</p>
          <img src="phase0/charts/phase0_boilerplate_by_category.png" alt="Boilerplate Overhead" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <!-- 5. DOM Structure Complexity -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">5. DOM Structural Elements</h3>
          <p class="text-xs text-slate-500 mb-3">Prevalence of headings, code, tables & lists.</p>
          <img src="phase0/charts/phase0_dom_complexity.png" alt="DOM Structural Elements" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <!-- 6. Technology Lengths -->
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">6. Document Lengths</h3>
          <p class="text-xs text-slate-500 mb-3">Average character length across technologies.</p>
          <img src="phase0/charts/phase0_tech_lengths.png" alt="Document Lengths" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>
      </div>
    </div>

    <!-- Section 2: Phase 1 Structured Corpus Analytics -->
    <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
      <div class="border-b border-slate-100 pb-3">
        <h2 class="text-lg font-bold text-slate-900">Phase 1: Structured Extraction Metrics</h2>
        <p class="text-xs text-slate-500">Output metrics from the multi-zone HTML parser.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">Category Distribution</h3>
          <img src="phase1/charts/phase1_category_distribution.png" alt="Phase 1 Category Distribution" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">Code Block Prevalence</h3>
          <img src="phase1/charts/phase1_code_prevalence.png" alt="Phase 1 Code Prevalence" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>

        <div class="bg-slate-50 border border-slate-200/80 rounded-xl p-4">
          <h3 class="text-sm font-bold text-slate-800 mb-1">Top Technologies Extracted</h3>
          <img src="phase1/charts/phase1_top_technologies.png" alt="Phase 1 Top Techs" class="rounded-lg w-full object-contain" onerror="this.style.display='none'"/>
        </div>
      </div>
    </div>

    <!-- Human-in-the-Loop Sign-off Decisions -->
    <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
      <h3 class="text-base font-bold text-slate-900 mb-2">Human-in-the-Loop Sign-off Decisions</h3>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm mt-3">
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-200/80">
          <span class="font-semibold text-slate-800">1. Body Length Threshold</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Exclude &lt; 200 chars to drop empty stubs while retaining API docs.</p>
        </div>
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-200/80">
          <span class="font-semibold text-slate-800">2. Language Filter</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Drop non-English docs (e.g. Japanese Oracle docs, Chinese docs).</p>
        </div>
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-200/80">
          <span class="font-semibold text-slate-800">3. Deduplication</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Collapse SimHash clusters (Hamming ≤ 3) to 1 representative document.</p>
        </div>
        <div class="p-4 bg-slate-50 rounded-lg border border-slate-200/80">
          <span class="font-semibold text-slate-800">4. BM25F Zones</span>
          <p class="text-slate-500 text-xs mt-1">Recommended: Zone weights across Title, Headings, Body, and Code blocks.</p>
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
      const nonEngPct = p0.exclusion_percentages?.non_english || 0;
      document.getElementById('kpi-lang').innerText = `${{nonEngPct}}%`;
    }}

    if (p0 && p0.dom_structure_stats) {{
      const codePct = p0.dom_structure_stats.pct_with_code || 0;
      document.getElementById('kpi-code').innerText = `${{codePct}}%`;
    }} else if (p1 && p1.code_statistics) {{
      const pct = p1.code_statistics.pct_docs_with_code || 0;
      document.getElementById('kpi-code').innerText = `${{pct}}%`;
    }}
  </script>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Generated enhanced interactive dashboard HTML at {output_html_path}")

