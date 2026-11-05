#!/usr/bin/env python3
"""Generate publication-ready evaluation charts for the IR benchmark."""

import csv
import os
from pathlib import Path

# Ensure writable matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "evaluation/results/metrics.csv"
CHARTS_DIR = ROOT / "evaluation/results/charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_overall_metrics():
    with METRICS_CSV.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("query_type") == "all"]

    labels_map = {
        ("classical", "bm25"): "Flat BM25",
        ("classical", "bm25f"): "Hierarchical BM25F",
        ("neural", "dense"): "Dense Neural",
        ("neural", "zones"): "Routed Cluster",
    }
    systems = []
    for r in rows:
        key = (r["engine"], r["variant"])
        if key in labels_map:
            systems.append((labels_map[key], r))

    metric_keys = [
        ("pooled_ndcg@5", "NDCG@5"),
        ("pooled_ndcg@10", "NDCG@10"),
        ("precision@5", "P@5"),
        ("precision@10", "P@10"),
        ("pooled_recall@10", "Recall@10"),
        ("pooled_MAP@10", "MAP@10"),
        ("MRR@10", "MRR@10"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_keys))
    width = 0.2
    colors = ["#2b5c8f", "#4b9cd3", "#d95f02", "#7570b3"]

    for i, (name, r) in enumerate(systems):
        values = [float(r[k]) for k, _ in metric_keys]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Overall Retrieval Effectiveness Comparison across 4 Paradigms", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title in metric_keys], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=11, loc="upper right")
    plt.tight_layout()
    out = CHARTS_DIR / "overall_metrics_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_per_query_comparison():
    with METRICS_CSV.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("query_type") != "all"]

    categories = sorted(list({r["query_type"] for r in rows}))
    systems = [
        ("classical", "bm25", "BM25", "#2b5c8f"),
        ("classical", "bm25f", "BM25F", "#4b9cd3"),
        ("neural", "dense", "Dense", "#d95f02"),
        ("neural", "zones", "Routed", "#7570b3"),
    ]

    data = {sys_name: [] for _, _, sys_name, _ in systems}
    for cat in categories:
        for eng, var, sys_name, _ in systems:
            match = [r for r in rows if r["query_type"] == cat and r["engine"] == eng and r["variant"] == var]
            score = float(match[0]["pooled_ndcg@10"]) if match else 0.0
            data[sys_name].append(score)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.2

    for i, (_, _, sys_name, color) in enumerate(systems):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[sys_name], width, label=sys_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("NDCG@10", fontsize=12, fontweight="bold")
    ax.set_title("NDCG@10 Breakdown by Technical Query Domain", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=11, loc="upper right")
    plt.tight_layout()
    out = CHARTS_DIR / "per_domain_ndcg10.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_latency_effectiveness_tradeoff():
    # Read NDCG@10 from metrics.csv
    ndcg_by_key = {}
    with METRICS_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("query_type") == "all":
                ndcg_by_key[(r["engine"], r["variant"])] = float(r.get("pooled_ndcg@10", 0.0))

    # Measured latencies from production runtime
    systems = [
        ("Flat BM25", ndcg_by_key.get(("classical", "bm25"), 0.7974), 5.3, "#2b5c8f", "o"),
        ("Hierarchical BM25F", ndcg_by_key.get(("classical", "bm25f"), 0.7987), 8.2, "#4b9cd3", "s"),
        ("Dense Neural (Full Scan)", ndcg_by_key.get(("neural", "dense"), 0.7521), 248.0, "#d95f02", "^"),
        ("Neural Routed-Cluster", ndcg_by_key.get(("neural", "zones"), 0.5859), 19.8, "#7570b3", "D"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, ndcg, lat, color, marker in systems:
        ax.scatter([lat], [ndcg], color=color, s=250, label=name, marker=marker, edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(
            f"{name}\n(NDCG: {ndcg:.3f}, {lat:.1f}ms)",
            (lat, ndcg),
            textcoords="offset points",
            xytext=(10, -15 if "Routed" in name else 5),
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Query Latency (ms) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_ylabel("NDCG@10", fontsize=12, fontweight="bold")
    ax.set_title("Efficiency vs. Effectiveness Pareto Frontier", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0.50, 0.90)
    ax.set_xlim(2, 500)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10, loc="lower right")
    plt.tight_layout()
    out = CHARTS_DIR / "latency_vs_effectiveness_pareto.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_overall_metrics()
    plot_per_query_comparison()
    plot_latency_effectiveness_tradeoff()
    print("All charts successfully generated!")

