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
    """1. Corpus Reduction & Index Footprint Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.3]})
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("#FCFCFD")
    ax2.set_facecolor("#FCFCFD")

    # Left: Raw Web Crawl vs. Filtered Production Corpus (GB & Document Counts)
    corpus_stages = [
        "Raw Web Crawl\n(79,268 HTML Files)",
        "Filtered Corpus\n(45,881 HTML Files)",
    ]
    corpus_gb = [18.6, 11.0]
    colors1 = [SLATE, NAVY]

    bars1 = ax1.bar(corpus_stages, corpus_gb, color=colors1, width=0.48, edgecolor="#1A202C", linewidth=0.8, alpha=0.92)
    labels1 = [
        "18.6 GB\n(79,268 docs)",
        "11.0 GB\n(45,881 docs)",
    ]
    for bar, txt in zip(bars1, labels1):
        ax1.annotate(
            txt,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2D3748",
        )
    ax1.set_title("Corpus Filtering Scale (GB & Document Count)", fontsize=11.0, fontweight="bold", color=NAVY, pad=10)
    ax1.set_ylabel("Storage on Disk (GB)", fontsize=9.5, fontweight="bold", color=SLATE)
    ax1.set_ylim(0, 23)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Classical vs Neural Pipeline Footprint (MB / GB)
    sys_components = [
        "Classical\n(JSON Text)",
        "Classical\n(Compressed)",
        "Qwen3 Vectors\n(4 Zones .npy)",
        "Total Neural Pipeline\n(Model+Windows+Vectors)",
    ]
    sys_mb = [1140.0, 95.0, 615.0, 2890.0]
    colors2 = [CRIMSON, TEAL, GOLD, PURPLE]

    bars2 = ax2.bar(sys_components, sys_mb, color=colors2, width=0.52, edgecolor="#1A202C", linewidth=0.8, alpha=0.92)
    for bar, val in zip(bars2, sys_mb):
        txt = f"{val/1024:.2f} GB" if val >= 1000 else f"{int(val)} MB"
        ax2.annotate(
            txt,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#2D3748",
        )
    ax2.set_title("Artifact Footprint (45.8k Docs: JSON vs. Packed vs. Neural)", fontsize=11.0, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylabel("Storage Footprint (MB)", fontsize=9.5, fontweight="bold", color=SLATE)
    ax2.set_ylim(0, 3500)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "index_size_breakdown.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_overall_effectiveness():
    """2. Multi-Metric Benchmark Comparison Across 5 Paradigms (Compact for Slide 12)."""
    fig, ax = plt.subplots(figsize=(12.0, 3.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    metrics = ["Graded NDCG@5", "Graded NDCG@10", "P@5", "P@10", "Recall@10", "MRR@10", "Binary NDCG"]

    data = {
        "Flat BM25": [0.7212, 0.8600, 0.7867, 0.7000, 0.5176, 0.9333, 0.9255],
        "BM25F": [0.7406, 0.8748, 0.7333, 0.6800, 0.5010, 0.9556, 0.9237],
        "Cluster-Routed Neural": [0.8042, 0.9088, 0.8667, 0.6867, 0.4987, 0.9667, 0.9446],
        "Qwen3 Dense": [0.7252, 0.8639, 0.8133, 0.7333, 0.5214, 0.9222, 0.9253],
        "Hybrid RRF": [0.7690, 0.9104, 0.8133, 0.7333, 0.5380, 1.0000, 0.9547],
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
            linewidth=0.5,
        )

    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=9.5, fontweight="bold", color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, 1.18)

    # Legend positioned cleanly below title with 70+ px gap (zero overlap)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E0",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fontsize=8.5,
    )
    ax.set_title("Retrieval Effectiveness Across 15 Benchmark Queries", fontsize=11.5, fontweight="bold", color=NAVY, pad=26)
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
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    depths = ["P@1", "P@3", "P@5", "P@10"]
    x = np.arange(len(depths))

    curves = {
        "Flat BM25": ([0.8667, 0.8222, 0.7867, 0.7000], SLATE, "o-"),
        "BM25F": ([0.9333, 0.8000, 0.7333, 0.6800], SKY_BLUE, "s-"),
        "Cluster-Routed Neural": ([0.9333, 0.8667, 0.8667, 0.6867], CRIMSON, "^-"),
        "Qwen3 Dense": ([0.8667, 0.8667, 0.8133, 0.7333], GOLD, "D-"),
        "Hybrid RRF": ([1.0000, 0.8889, 0.8133, 0.7333], PURPLE, "P-"),
    }

    for name, (vals, color, fmt) in curves.items():
        linewidth = 2.2 if name == "Hybrid RRF" else 1.6
        markersize = 9 if name == "Hybrid RRF" else 6
        ax.plot(x, vals, fmt, label=name, color=color, linewidth=linewidth, markersize=markersize, zorder=5)

    ax.set_title("Precision Retention Across Rank Cutoffs (P@1 → P@10)", fontsize=11.5, fontweight="bold", color=NAVY, pad=12)
    ax.set_ylabel("Precision Score", fontsize=9.5, fontweight="bold", color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.60, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="lower left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "precision_depth_curve.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_latency_pareto():
    """4. Latency vs. Graded NDCG@10 (Simplified title, both latency & NDCG on points, spacious legend)."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    points = [
        ("Flat BM25", 31.0, 0.8600, NAVY, "o"),
        ("BM25F", 43.5, 0.8748, SKY_BLUE, "s"),
        ("Cluster-Routed", 23.3, 0.9088, CRIMSON, "^"),
        ("Qwen3 Dense", 53.7, 0.8639, GOLD, "D"),
        ("Hybrid RRF", 66.8, 0.9104, PURPLE, "P"),
    ]

    for name, lat, ndcg, color, marker in points:
        ax.scatter(lat, ndcg, color=color, s=140, marker=marker, edgecolors="black", linewidth=1.0, zorder=5, label=name)

    # Point labels with BOTH latency and NDCG (No stars, clean text)
    ax.annotate("Flat BM25\n(31.0 ms, 0.8600)", (31.0, 0.8600), xytext=(8, -16), textcoords="offset points", fontsize=8.0, fontweight="bold", color=SLATE)
    ax.annotate("BM25F\n(43.5 ms, 0.8748)", (43.5, 0.8748), xytext=(8, -16), textcoords="offset points", fontsize=8.0, fontweight="bold", color=SLATE)
    ax.annotate("Cluster-Routed\n(23.3 ms, 0.9088)", (23.3, 0.9088), xytext=(-95, 10), textcoords="offset points", fontsize=8.0, fontweight="bold", color=CRIMSON)
    ax.annotate("Qwen3 Dense\n(53.7 ms, 0.8639)", (53.7, 0.8639), xytext=(8, -16), textcoords="offset points", fontsize=8.0, fontweight="bold", color=SLATE)
    ax.annotate("Hybrid RRF\n(66.8 ms, 0.9104)", (66.8, 0.9104), xytext=(-85, 10), textcoords="offset points", fontsize=8.0, fontweight="bold", color=PURPLE)

    ax.set_xscale("log")
    ax.set_xlim(14, 130)
    ax.set_ylim(0.835, 0.935)
    ax.set_title("Cluster-Routed gives highest efficiency (23.3 ms), followed by Hybrid RRF (0.9104)", fontsize=10.5, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Query Latency (ms) — Logarithmic Scale", fontsize=9.0, fontweight="bold", color=SLATE)
    ax.set_ylabel("Graded NDCG@10", fontsize=9.0, fontweight="bold", color=SLATE)
    ax.grid(True, linestyle="--", alpha=0.35, color="#A0AEC0", which="both")
    # Legend at center left so it is far away from points and ticks
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="center left", bbox_to_anchor=(0.02, 0.45), fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "latency_throughput_pareto.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_cluster_category_alignment():
    """5. K-Means++ Domain Clusters with Soft Multi-Membership Expansion (Stacked Horizontal Bar)."""
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

    primary_docs = np.array([3501, 5207, 6579, 5376, 5282, 5008, 4010, 10918])
    secondary_docs = np.array([502, 773, 1545, 1720, 2767, 1363, 6, 3816])
    total_docs = primary_docs + secondary_docs
    dominant_share = [93.4, 81.2, 74.8, 68.5, 62.1, 58.7, 52.3, 46.2]

    y = np.arange(len(clusters))

    # Stacked horizontal bar: Primary (Navy) + Secondary Expansion (Gold with hatch)
    b1 = ax.barh(y, primary_docs, color=NAVY, height=0.58, edgecolor="black", linewidth=0.7, alpha=0.92, label="Primary Cluster Documents")
    b2 = ax.barh(y, secondary_docs, left=primary_docs, color=GOLD, height=0.58, edgecolor="black", linewidth=0.7, hatch="//", alpha=0.92, label="Secondary Boundary Documents (Δ ≤ 0.03)")

    for idx, (p_count, s_count, tot, share) in enumerate(zip(primary_docs, secondary_docs, total_docs, dominant_share)):
        pct_inc = (s_count / p_count) * 100
        ax.annotate(
            f"{tot:,} docs (+{pct_inc:.1f}% secondary)  ·  {share}% domain purity",
            xy=(tot, idx),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color=SLATE,
        )

    ax.set_title("K-Means++ (k=8) Domain Clustering & Soft Multi-Membership Expansion (Δ ≤ 0.03)", fontsize=11.5, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Number of Indexed Documents in Cluster", fontsize=9.5, fontweight="bold", color=SLATE)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 19000)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#A0AEC0")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = OUTPUT_DIR / "cluster_category_alignment.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    plot_storage_footprint()
    plot_overall_effectiveness()
    plot_precision_depth_curve()
    plot_latency_pareto()
    plot_cluster_category_alignment()
    print("All presentation charts generated cleanly in presentation/charts/")
