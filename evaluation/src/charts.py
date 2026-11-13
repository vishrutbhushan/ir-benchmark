"""Generate benchmark charts from the saved evaluation metrics."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ir-benchmark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import load_pipeline_config, resolve_path

SYSTEMS = (
    ("classical", "bm25", "BM25", "#24316f", "o"),
    ("classical", "bm25f", "BM25F", "#4c9ed9", "s"),
    ("neural", "dense", "Dense neural", "#e49a23", "D"),
    ("neural", "zones", "Zone-weighted neural", "#7655a7", "v"),
    ("neural", "routed", "Cluster-routed neural", "#26a196", "^"),
    ("hybrid", "rrf", "Hybrid RRF", "#e1482f", "P"),
)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(per_query, per_query_graded, query_ids):
    by_query = {row["query_id"]: row for row in per_query}
    by_graded = {row["query_id"]: row for row in per_query_graded}
    rows = [by_query[qid] for qid in query_ids]
    grades = [by_graded[qid] for qid in query_ids]
    return {
        "Binary NDCG@5": np.mean([float(row["pooled_ndcg@5"]) for row in rows]),
        "Binary NDCG@10": np.mean([float(row["pooled_ndcg@10"]) for row in rows]),
        "Graded NDCG@10": np.mean([float(row["graded_ndcg10"]) for row in grades]),
        "P@5": np.mean([float(row["precision@5"]) for row in rows]),
        "P@10": np.mean([float(row["precision@10"]) for row in rows]),
        "Recall@10": np.mean([float(row["pooled_recall@10"]) for row in rows]),
        "MRR@10": np.mean([float(row["rr@10"]) for row in rows]),
    }


def plot_metrics(rows_by_system, title, destination):
    labels = list(next(iter(rows_by_system.values())))
    x = np.arange(len(labels))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11.9, 4.9), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfd")
    for i, (engine, variant, name, color, _) in enumerate(SYSTEMS):
        values = [rows_by_system[(engine, variant)][label] for label in labels]
        ax.bar(x + (i - 2.5) * width, values, width, label=name, color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=11.5, color="#211d70", pad=12)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_precision(per_query, query_ids, destination):
    lookup = defaultdict(dict)
    for row in per_query:
        lookup[(row["engine"], row["variant"])][row["query_id"]] = row
    x = (1, 3, 5, 10)
    fig, ax = plt.subplots(figsize=(9.4, 4.7), dpi=300)
    for engine, variant, name, color, marker in SYSTEMS:
        rows = [lookup[(engine, variant)][qid] for qid in query_ids]
        values = [np.mean([float(row[f"precision@{k}"]) for row in rows]) for k in x]
        ax.plot(x, values, marker=marker, color=color, linewidth=1.8, markersize=6, label=name)
    ax.set_title("Precision at Rank Cutoffs (20 queries)", color="#211d70", pad=12)
    ax.set_xlabel("Rank cutoff")
    ax.set_ylabel("Precision")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.04)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, loc="lower left", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_latency(summary, manifests_path, latencies, destination, cutoff):
    manifest = json.loads(Path(manifests_path).read_text(encoding="utf-8"))
    latency_map = {
        ("classical", "bm25"): float(latencies["bm25"]),
        ("classical", "bm25f"): float(latencies["bm25f"]),
        ("hybrid", "rrf"): float(latencies["rrf"]),
    }
    latency_map.update({("neural", item["variant"]): float(item["retrieval_median_ms"]) for item in manifest})
    metric = f"pooled_ndcg@{cutoff}"
    lookup = {(row["engine"], row["variant"]): row for row in summary if row["query_type"] == "all"}
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=300)
    offsets = {
        "bm25": (8, -18), "bm25f": (-55, -35), "dense": (8, 14),
        "zones": (8, 16), "routed": (8, -26), "rrf": (8, 8),
    }
    for engine, variant, name, color, marker in SYSTEMS:
        x = latency_map[(engine, variant)]
        y = float(lookup[(engine, variant)][metric])
        ax.scatter(x, y, s=110, color=color, marker=marker, label=name, edgecolors="white", linewidths=0.8)
        ax.annotate(f"{name}\n({x:.1f} ms, {y:.3f})", (x, y), xytext=offsets[variant], textcoords="offset points", fontsize=8, color=color)
    ax.set_title(f"Latency–effectiveness trade-off (Binary NDCG@{cutoff})", color="#211d70", pad=12)
    ax.set_xlabel("Query latency (ms)")
    ax.set_ylabel(f"Binary NDCG@{cutoff}")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def generate_charts(config_path=None):
    config = load_pipeline_config(config_path)
    results = resolve_path(config["paths"]["evaluation_results_dir"])
    charts = results / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    summary = read_csv(results / "metrics.csv")
    per_query = read_csv(results / "per_query.csv")
    per_query_graded = read_csv(results / "per_query_graded.csv")
    query_ids = [
        json.loads(line)["query_id"]
        for line in resolve_path(config["paths"]["queries"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = config["evaluation"]["query_groups"]
    query_ranges = {
        "overall": query_ids,
        "keyword": query_ids[groups["keyword"]["start"] - 1:groups["keyword"]["end"]],
        "natural_language": query_ids[groups["natural_language"]["start"] - 1:groups["natural_language"]["end"]],
    }
    titles = {
        "overall": "Overall retrieval metrics (20 queries)",
        "keyword": "Keyword and technical syntax queries (Q01–Q10)",
        "natural_language": "Natural-language queries (Q11–Q20)",
    }
    filenames = {
        "overall": "overall_metrics_comparison.png",
        "keyword": "overall_metrics_keyword.png",
        "natural_language": "overall_metrics_nl.png",
    }
    for group, ids in query_ranges.items():
        rows_by_system = {}
        for engine, variant, *_ in SYSTEMS:
            pq = [r for r in per_query if (r["engine"], r["variant"]) == (engine, variant)]
            pg = [r for r in per_query_graded if (r["engine"], r["variant"]) == (engine, variant)]
            rows_by_system[(engine, variant)] = aggregate(pq, pg, ids)
        plot_metrics(rows_by_system, titles[group], charts / filenames[group])
    plot_precision(per_query, query_ids, charts / "precision_depth_curve.png")
    manifests = results / "neural/manifest.json"
    latency = config["evaluation"]["reported_latency_ms"]
    plot_latency(summary, manifests, latency, charts / "latency_vs_effectiveness_at5.png", 5)
    plot_latency(summary, manifests, latency, charts / "latency_vs_effectiveness_pareto.png", 10)
    return charts
