#!/usr/bin/env python3
"""
Generate publication-quality presentation charts for the IR Benchmark PPTX.

Theme: BITS Pilani Palette
- Primary Navy:   #211D70
- Gold / Amber:   #EA9830
- Crimson Red:    #EA3F26
- Sky Blue:       #72C3E8
- Slate Dark:     #2D3748
- Muted Grey:     #718096
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("presentation/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Brand Colors
NAVY = "#211D70"
GOLD = "#EA9830"
CRIMSON = "#EA3F26"
SKY_BLUE = "#72C3E8"
TEAL = "#20B2AA"
PURPLE = "#6B46C1"
SLATE = "#2D3748"
BG_LIGHT = "#F8F9FA"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def plot_storage_footprint():
    """1. Index Storage & Disk Breakdown."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    components = [
        "Raw HTML Corpus\n(WARC Extracted)",
        "Processed Corpus\n(Clean SQLite)",
        "Classical Index\n(BM25 Inverted JSON)",
        "Dense Vectors\n(4-Zone MiniLM)",
        "Cluster Centroids\n& Metadata",
    ]
    sizes_mb = [1530.0, 1120.0, 1114.0, 615.0, 0.07]
    colors = [SLATE, NAVY, SKY_BLUE, GOLD, CRIMSON]

    bars = ax.bar(components, sizes_mb, color=colors, width=0.55, edgecolor="#1A202C", linewidth=0.8, alpha=0.92)

    for bar, size in zip(bars, sizes_mb):
        yval = bar.get_height()
        if size >= 1000:
            txt = f"{size/1024:.2f} GB\n({int(size)} MB)"
        elif size >= 1:
            txt = f"{int(size)} MB"
        else:
            txt = f"{size*1024:.0f} KB"
        ax.annotate(
            txt,
            xy=(bar.get_x() + bar.get_width() / 2, yval),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2D3748",
        )

    ax.set_title("Storage Footprint Across Pipeline Artifacts (45,881 Documents)", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_ylabel("Storage Footprint (MB)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_ylim(0, 1850)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "index_size_breakdown.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_latency_pareto():
    """2. Latency vs. Effectiveness Pareto Frontier."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    points = [
        ("Flat BM25", 5.3, 0.6727, NAVY, "o", "5.3 ms\n188.7 QPS"),
        ("Hierarchical BM25F", 8.2, 0.7089, SKY_BLUE, "s", "8.2 ms\n122.0 QPS"),
        ("Cluster Routed Neural", 19.8, 0.6335, CRIMSON, "^", "19.8 ms (75% pruned)\n50.5 QPS"),
        ("Dense Full Scan", 248.0, 0.8036, GOLD, "D", "248.0 ms (45.8k scan)\n4.0 QPS"),
        ("Hybrid RRF (Dense+BM25F)", 24.5, 0.8235, PURPLE, "*", "Optimal Tradeoff\nNDCG: 0.8235, MRR: 1.0"),
    ]

    for name, lat, ndcg, color, marker, note in points:
        size = 280 if marker == "*" else 160
        ax.scatter(lat, ndcg, color=color, s=size, marker=marker, edgecolors="black", linewidth=1.2, zorder=5, label=name)

    # Annotate points
    ax.annotate("Flat BM25\n(5.3 ms)", (5.3, 0.6727), xytext=(5, -18), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Hierarchical BM25F\n(8.2 ms)", (8.2, 0.7089), xytext=(8, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Cluster Routed\n(19.8 ms)", (19.8, 0.6335), xytext=(10, -15), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Dense Full Scan\n(248.0 ms)", (248.0, 0.8036), xytext=(-85, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("★ Hybrid RRF (k=20)\n(Best Quality: 0.8235)", (24.5, 0.8235), xytext=(-35, 12), textcoords="offset points", fontsize=9.5, fontweight="bold", color=PURPLE)

    # Pareto curve connection
    pareto_x = [5.3, 8.2, 24.5, 248.0]
    pareto_y = [0.6727, 0.7089, 0.8235, 0.8036]
    ax.plot([5.3, 8.2, 24.5], [0.6727, 0.7089, 0.8235], linestyle=":", color=PURPLE, alpha=0.7, linewidth=1.8, label="Empirical Pareto Frontier")

    ax.set_xscale("log")
    ax.set_xlim(3, 400)
    ax.set_ylim(0.58, 0.88)
    ax.set_title("Retrieval Latency vs. Effectiveness (NDCG@10)", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_xlabel("Query Latency (ms) — Logarithmic Scale", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_ylabel("Effectiveness (NDCG@10)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.grid(True, linestyle="--", alpha=0.35, color="#A0AEC0", which="both")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="lower right", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "latency_throughput_pareto.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_effectiveness_comparison():
    """3. Overall Retrieval Effectiveness across Paradigms."""
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    paradigms = [
        "BM25\n(Flat Lexical)",
        "BM25F\n(Multi-Field)",
        "Neural Routed\n(Cluster Top-2)",
        "Neural Dense\n(Semantic)",
        "Hybrid RRF\n(Dense + BM25F)",
    ]

    ndcg10 = [0.6727, 0.7089, 0.6335, 0.8036, 0.8235]
    p5 = [0.7067, 0.7333, 0.6800, 0.8133, 0.8533]
    mrr = [0.8667, 0.8889, 0.9000, 0.9667, 1.0000]

    x = np.arange(len(paradigms))
    width = 0.25

    rects1 = ax.bar(x - width, ndcg10, width, label="NDCG@10", color=NAVY, alpha=0.9, edgecolor="black", linewidth=0.6)
    rects2 = ax.bar(x, p5, width, label="Precision@5", color=GOLD, alpha=0.9, edgecolor="black", linewidth=0.6)
    rects3 = ax.bar(x + width, mrr, width, label="MRR@10", color=SKY_BLUE, alpha=0.9, edgecolor="black", linewidth=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_title("Benchmarked Retrieval Effectiveness across 5 Architectural Variants", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_ylabel("Metric Score (0.0 – 1.0)", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="upper left", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "overall_effectiveness_hybrid.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_cluster_category_alignment():
    """4. Unsupervised Neural Clustering vs. Engineering Domains."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    clusters = [
        "C4: DevOps / VCS\n(GitHub, GitLab, Git)",
        "C3: AI / ML Libs\n(PyTorch, CUDA, Ray, HF)",
        "C7: Databases & SQL\n(PostgreSQL, MySQL, ClickHouse)",
        "C6: Observability / K8s\n(Grafana, Kubernetes)",
        "C0: OS & Networking\n(Linux Kernel, Envoy, OpenSSL)",
        "C2: Frontend & Build\n(Maven, Gradle, Angular, MDN)",
        "C5: Web & Cloud VM\n(AWS, Apache, Mongo)",
        "C1: Cloud Infra & Search\n(Azure, Oracle, Elastic)",
    ]

    doc_counts = [3439, 5156, 6575, 5318, 5499, 5121, 4010, 10763]
    dominant_share = [93.4, 81.2, 74.8, 68.5, 62.1, 58.7, 52.3, 46.2]
    colors = [NAVY, GOLD, CRIMSON, SKY_BLUE, TEAL, PURPLE, SLATE, "#D69E2E"]

    y = np.arange(len(clusters))
    bars = ax.barh(y, doc_counts, color=colors, height=0.6, edgecolor="black", linewidth=0.7, alpha=0.9)

    for bar, count, share in zip(bars, doc_counts, dominant_share):
        xval = bar.get_width()
        ax.annotate(
            f"{count:,} docs  ({share}% domain purity)",
            xy=(xval, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=SLATE,
        )

    ax.set_title("Unsupervised K-Means++ (k=8) Discovers Ground-Truth Engineering Domains", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_xlabel("Number of Indexed Documents in Cluster", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=9.0, fontweight="bold")
    ax.set_xlim(0, 13500)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "cluster_category_alignment.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_cache_speedup():
    """5. LRU Query Cache Hit vs. Cold Miss Latency."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    engines = [
        "BM25 (Flat)",
        "BM25F (Fields)",
        "Cluster Routed",
        "Hybrid RRF",
        "Dense (Full Scan)",
    ]

    cold_ms = [5.3, 8.2, 19.8, 24.5, 248.0]
    cached_ms = [0.04, 0.04, 0.04, 0.04, 0.04]
    speedups = [132, 205, 495, 612, 6200]

    y = np.arange(len(engines))
    height = 0.35

    rects1 = ax.barh(y - height/2, cold_ms, height, label="Cold Miss (Full Index Scoring)", color=CRIMSON, alpha=0.88, edgecolor="black", linewidth=0.6)
    rects2 = ax.barh(y + height/2, cached_ms, height, label="LRU Cache Hit (<0.05 ms)", color=TEAL, alpha=0.88, edgecolor="black", linewidth=0.6)

    for bar, speedup, cold in zip(rects1, speedups, cold_ms):
        ax.annotate(
            f"{cold:.1f} ms  →  {speedup:,}× speedup",
            xy=(cold, bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=SLATE,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.01, 800)
    ax.set_title("Query-Result Caching Latency Acceleration (Cold Execution vs. Cache Hit)", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_xlabel("Response Latency (ms) — Logarithmic Scale", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_yticks(y)
    ax.set_yticklabels(engines, fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#A0AEC0", which="both")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="lower right", fontsize=9.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "cache_latency_speedup.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    plot_storage_footprint()
    plot_latency_pareto()
    plot_effectiveness_comparison()
    plot_cluster_category_alignment()
    plot_cache_speedup()
    print("All presentation charts generated successfully in presentation/charts/")
