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
    """1. Comprehensive Storage & Footprint Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.1, 1.3]})
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("#FCFCFD")
    ax2.set_facecolor("#FCFCFD")

    # Left: Corpus Progression (GB)
    corpus_stages = [
        "Raw Web Crawl\n(79k HTML Files)",
        "Extracted HTML\n(data/00_corpus)",
        "Filtered DB\n(Clean SQLite)",
    ]
    corpus_gb = [18.6, 11.0, 1.12]
    colors1 = [SLATE, NAVY, SKY_BLUE]

    bars1 = ax1.bar(corpus_stages, corpus_gb, color=colors1, width=0.55, edgecolor="#1A202C", linewidth=0.8, alpha=0.92)
    for bar, val in zip(bars1, corpus_gb):
        ax1.annotate(
            f"{val:.2f} GB" if val < 10 else f"{val:.1f} GB",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2D3748",
        )
    ax1.set_title("Corpus Filtering & Ingestion Scale", fontsize=11.5, fontweight="bold", color=NAVY, pad=12)
    ax1.set_ylabel("Storage Footprint (GB)", fontsize=10, fontweight="bold", color=SLATE)
    ax1.set_ylim(0, 22)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Classical vs Neural Pipeline Footprint (MB / GB)
    sys_components = [
        "Classical Index\n(Uncompressed JSON)",
        "Classical Index\n(Standard VByte/Lucene Est.)",
        "Neural Vectors Only\n(4 Zones .npy)",
        "Total Neural Pipeline\n(Model+Windows+Vectors+Clusters)",
    ]
    sys_mb = [1140.0, 95.0, 615.0, 2890.0]
    colors2 = [CRIMSON, TEAL, GOLD, PURPLE]

    bars2 = ax2.bar(sys_components, sys_mb, color=colors2, width=0.55, edgecolor="#1A202C", linewidth=0.8, alpha=0.92)
    for bar, val in zip(bars2, sys_mb):
        txt = f"{val/1024:.2f} GB" if val >= 1000 else f"{int(val)} MB"
        ax2.annotate(
            txt,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2D3748",
        )
    ax2.set_title("Classical vs. Neural Artifact Footprint (45.8k Docs)", fontsize=11.5, fontweight="bold", color=NAVY, pad=12)
    ax2.set_ylabel("Storage Footprint (MB)", fontsize=10, fontweight="bold", color=SLATE)
    ax2.set_ylim(0, 3400)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "index_size_breakdown.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_overall_effectiveness():
    """2. Complete Multi-Metric Benchmark Comparison Across 5 Paradigms."""
    fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    metrics = ["NDCG@5", "NDCG@10", "P@5", "P@10", "Recall@10", "MAP@10", "MRR@10"]

    # Values for the 5 systems across all 7 metrics
    data = {
        "Flat BM25": [0.7188, 0.6727, 0.7067, 0.6400, 0.4643, 0.3978, 0.8667],
        "BM25F (Hierarchical)": [0.7503, 0.7089, 0.7333, 0.6800, 0.4870, 0.4124, 0.8889],
        "Neural Routed (k=8)": [0.7181, 0.6335, 0.6800, 0.5733, 0.3800, 0.3142, 0.9000],
        "Dense (Full Scan)": [0.8420, 0.8036, 0.8133, 0.7667, 0.5226, 0.4634, 0.9667],
        "★ Hybrid RRF (k=20)": [0.8931, 0.8235, 0.8533, 0.7667, 0.5226, 0.4864, 1.0000],
    }

    colors = [SLATE, SKY_BLUE, CRIMSON, GOLD, PURPLE]
    x = np.arange(len(metrics))
    num_systems = len(data)
    width = 0.15

    for i, (name, values) in enumerate(data.items()):
        offset = (i - (num_systems - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            values,
            width,
            label=name,
            color=colors[i],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_title("Comprehensive Retrieval Effectiveness Across 5 Architectures (All Primary Metrics)", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold", color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="upper right", fontsize=9.5, ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "overall_effectiveness_hybrid.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_precision_depth_curve():
    """3. Precision Decay Across Rank Depth (P@1, P@3, P@5, P@10)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    depths = ["P@1", "P@3", "P@5", "P@10"]
    x = np.arange(len(depths))

    curves = {
        "Flat BM25": ([0.7333, 0.7333, 0.7067, 0.6400], SLATE, "o-"),
        "BM25F (Fields)": ([0.8000, 0.8000, 0.7333, 0.6800], SKY_BLUE, "s-"),
        "Neural Routed": ([0.8000, 0.7556, 0.6800, 0.5733], CRIMSON, "^-"),
        "Dense Full Scan": ([0.9333, 0.8667, 0.8133, 0.7667], GOLD, "D-"),
        "★ Hybrid RRF": ([1.0000, 0.9333, 0.8533, 0.7667], PURPLE, "*-"),
    }

    for name, (vals, color, fmt) in curves.items():
        linewidth = 2.5 if "★" in name else 1.8
        markersize = 10 if "★" in name else 7
        ax.plot(x, vals, fmt, label=name, color=color, linewidth=linewidth, markersize=markersize, zorder=5)

    ax.set_title("Precision Retention Across Rank Cutoffs (P@1 → P@10)", fontsize=13, fontweight="bold", color=NAVY, pad=15)
    ax.set_ylabel("Precision Score", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.50, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="lower left", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "precision_depth_curve.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_latency_pareto():
    """4. Latency vs. Effectiveness Pareto Frontier."""
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

    ax.annotate("Flat BM25\n(5.3 ms)", (5.3, 0.6727), xytext=(5, -18), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Hierarchical BM25F\n(8.2 ms)", (8.2, 0.7089), xytext=(8, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Cluster Routed\n(19.8 ms)", (19.8, 0.6335), xytext=(10, -15), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("Dense Full Scan\n(248.0 ms)", (248.0, 0.8036), xytext=(-85, 10), textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.annotate("★ Hybrid RRF (k=20)\n(Best Quality: 0.8235)", (24.5, 0.8235), xytext=(-35, 12), textcoords="offset points", fontsize=9.5, fontweight="bold", color=PURPLE)

    pareto_x = [5.3, 8.2, 24.5]
    pareto_y = [0.6727, 0.7089, 0.8235]
    ax.plot(pareto_x, pareto_y, linestyle=":", color=PURPLE, alpha=0.7, linewidth=1.8, label="Empirical Pareto Frontier")

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


def plot_cluster_category_alignment():
    """5. Unsupervised Neural Clustering vs. Engineering Domains."""
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
    """6. LRU Query Cache Hit vs. Cold Miss Latency."""
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
    plot_overall_effectiveness()
    plot_precision_depth_curve()
    plot_latency_pareto()
    plot_cluster_category_alignment()
    plot_cache_speedup()
    print("All presentation charts updated successfully in presentation/charts/")
