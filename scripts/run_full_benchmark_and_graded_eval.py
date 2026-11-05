#!/usr/bin/env python3
"""
Full Benchmark Execution & Graded IR Evaluation:
1. Runs retrieval on all 15 queries across all systems:
   - Classical: BM25, BM25F
   - Neural: Dense, Neural Zones (learned weights), Cluster-Routed (Soft Delta=0.03)
   - Hybrid: Hybrid RRF
2. Saves result snapshots to evaluation/results/.
3. Identifies unjudged candidate document pairs and queries Ollama (1 worker, think=False).
4. Calculates Binary Metrics and Graded NDCG@5, Graded NDCG@10 (2^g - 1).
5. Exports evaluation/results/metrics.csv and prints the benchmark table.
"""

import csv
import json
import math
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "evaluation/queries/final_queries.jsonl"
JUDGMENTS_PATH = ROOT / "evaluation/data/judgments.csv"
ENSEMBLE_PATH = ROOT / "evaluation/data/judgments_ensemble.csv"
DB_PATH = ROOT / "filter/outputs/stage_04_filtering/corpus.sqlite"

OLLAMA_URL = "http://172.23.64.1:11434/api/chat"
MODEL = "gemma4:12b"

COLUMNS = [
    "engine",
    "variant",
    "query_id",
    "query_text",
    "rank",
    "doc_id",
    "title",
    "url",
    "technology",
    "categories",
    "score",
]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def call_gemma_grade(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 50, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    text = data.get("message", {}).get("content", "").strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        grade = int(parsed.get("grade", 0))
        return min(max(grade, 0), 2)
    except Exception:
        if '"grade": 2' in text or '"grade":2' in text:
            return 2
        elif '"grade": 1' in text or '"grade":1' in text:
            return 1
        return 0


def rrf_fuse(runs_list, k=60):
    rrf_scores = defaultdict(lambda: defaultdict(float))
    doc_meta = {}
    for run in runs_list:
        for r in run:
            qid = r["query_id"]
            did = r["doc_id"]
            rank = int(r["rank"])
            rrf_scores[qid][did] += 1.0 / (k + rank)
            doc_meta[did] = r

    fused_rows = []
    for qid in sorted(rrf_scores.keys()):
        sorted_docs = sorted(rrf_scores[qid].items(), key=lambda x: -x[1])[:10]
        for rank, (did, score) in enumerate(sorted_docs, 1):
            orig = doc_meta[did]
            fused_rows.append({
                "engine": "hybrid",
                "variant": "rrf",
                "query_id": qid,
                "query_text": orig["query_text"],
                "rank": rank,
                "doc_id": did,
                "title": orig["title"],
                "url": orig["url"],
                "technology": orig["technology"],
                "categories": orig["categories"],
                "score": f"{score:.6f}",
            })
    return fused_rows


def compute_graded_ndcg(grades, k=10):
    values = list(grades[:k])
    dcg = sum((2**int(g) - 1) / math.log2(rank + 1) for rank, g in enumerate(values, 1))
    ideal = sorted(grades, reverse=True)[:k]
    ideal_dcg = sum((2**int(g) - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_binary_ndcg(binaries, k=10):
    values = list(binaries[:k])
    dcg = sum((int(b) == 1) / math.log2(rank + 1) for rank, b in enumerate(values, 1))
    ideal = sorted(binaries, reverse=True)[:k]
    ideal_dcg = sum((int(b) == 1) / math.log2(rank + 1) for rank, b in enumerate(ideal, 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    print("=== [Step 1: Running Retrieval on All 15 Queries Across 6 Systems] ===", flush=True)
    queries = [
        json.loads(line)
        for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(queries)} queries.", flush=True)

    from run_search_ui import SearchRuntime

    runtime = SearchRuntime(
        ROOT / "neural/outputs",
        ROOT / "data/00_corpus",
        ROOT / "filter/outputs/stage_05_archive/technical_web_corpus.zip",
    )

    # 1. Classical Runs
    print("\n[1/3] Running Classical Engines (BM25, BM25F)...", flush=True)
    runtime._load_classical()
    bm25_rows, bm25f_rows = [], []
    for q in queries:
        qid, qtext = q["query_id"], q["query_text"]
        for r in runtime.search("bm25", qtext)["results"][:10]:
            bm25_rows.append({
                "engine": "classical",
                "variant": "bm25",
                "query_id": qid,
                "query_text": qtext,
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "url": r["url"],
                "technology": r.get("technology", ""),
                "categories": json.dumps(r.get("categories", [])),
                "score": f"{r['score']:.4f}",
            })
        for r in runtime.search("bm25f", qtext)["results"][:10]:
            bm25f_rows.append({
                "engine": "classical",
                "variant": "bm25f",
                "query_id": qid,
                "query_text": qtext,
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "url": r["url"],
                "technology": r.get("technology", ""),
                "categories": json.dumps(r.get("categories", [])),
                "score": f"{r['score']:.4f}",
            })

    write_csv(ROOT / "evaluation/results/classical/bm25.csv", bm25_rows)
    write_csv(ROOT / "evaluation/results/classical/bm25f.csv", bm25f_rows)

    # 2. Neural Runs
    print("\n[2/3] Running Neural Engines (Dense, Zones, Soft-Routed)...", flush=True)
    runtime._load_neural()
    dense_rows, zones_rows, routed_rows = [], [], []
    for q in queries:
        qid, qtext = q["query_id"], q["query_text"]
        # Dense
        for r in runtime.search("dense", qtext)["results"][:10]:
            dense_rows.append({
                "engine": "neural",
                "variant": "dense",
                "query_id": qid,
                "query_text": qtext,
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "url": r["url"],
                "technology": r.get("technology", ""),
                "categories": json.dumps(r.get("categories", [])),
                "score": f"{r['score']:.4f}",
            })
        # Neural Zones
        for r in runtime.search("neural_zones", qtext)["results"][:10]:
            zones_rows.append({
                "engine": "neural",
                "variant": "zones",
                "query_id": qid,
                "query_text": qtext,
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "url": r["url"],
                "technology": r.get("technology", ""),
                "categories": json.dumps(r.get("categories", [])),
                "score": f"{r['score']:.4f}",
            })
        # Cluster-Routed (Soft)
        for r in runtime.neural.search(qtext, mode="routed-cluster")["results"][:10]:
            routed_rows.append({
                "engine": "neural",
                "variant": "cluster_routed",
                "query_id": qid,
                "query_text": qtext,
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "url": r["url"],
                "technology": r.get("technology", ""),
                "categories": json.dumps(r.get("categories", [])),
                "score": f"{r['score']:.4f}",
            })

    write_csv(ROOT / "evaluation/results/neural/dense.csv", dense_rows)
    write_csv(ROOT / "evaluation/results/neural/zones.csv", zones_rows)
    write_csv(ROOT / "evaluation/results/neural/routed.csv", routed_rows)

    # 3. Hybrid Run
    print("\n[3/3] Running Hybrid RRF (BM25F + Cluster-Routed Neural Soft)...", flush=True)
    rrf_rows = rrf_fuse([bm25f_rows, routed_rows], k=60)
    write_csv(ROOT / "evaluation/results/hybrid/rrf.csv", rrf_rows)

    all_runs = {
        ("classical", "bm25"): bm25_rows,
        ("classical", "bm25f"): bm25f_rows,
        ("neural", "dense"): dense_rows,
        ("neural", "zones"): zones_rows,
        ("neural", "cluster_routed"): routed_rows,
        ("hybrid", "rrf"): rrf_rows,
    }

    # Step 2: Pool & Check Judgments
    print("\n=== [Step 2: Checking Judgment Pool for All Retrieved Documents] ===", flush=True)
    existing_binary = {}
    existing_graded = {}

    if ENSEMBLE_PATH.exists():
        with open(ENSEMBLE_PATH, mode="r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                qid, did = r["query_id"], r["doc_id"]
                grade = int(r.get("consensus_grade", 0))
                existing_graded[(qid, did)] = grade
                existing_binary[(qid, did)] = 1 if grade >= 1 else 0

    if JUDGMENTS_PATH.exists():
        with open(JUDGMENTS_PATH, mode="r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                qid, did = r["query_id"], r["doc_id"]
                if (qid, did) not in existing_binary and r["relevant"]:
                    rel = int(r["relevant"])
                    existing_binary[(qid, did)] = rel
                    if (qid, did) not in existing_graded:
                        existing_graded[(qid, did)] = 2 if rel == 1 else 0

    all_pool_pairs = set()
    for run in all_runs.values():
        for r in run:
            all_pool_pairs.add((r["query_id"], r["doc_id"]))

    unjudged_pairs = [pair for pair in all_pool_pairs if pair not in existing_graded]
    print(f"Total Unique Retrieved (Query, Doc) Pairs: {len(all_pool_pairs)}")
    print(f"Existing Judged Pairs                     : {len(all_pool_pairs) - len(unjudged_pairs)}")
    print(f"Unjudged Pairs Requiring LLM Evaluation   : {len(unjudged_pairs)}")

    if unjudged_pairs:
        print("\n=== [Step 3: Sequential Judging of New Candidate Pairs (1 Worker)] ===", flush=True)
        con = sqlite3.connect(DB_PATH)
        doc_cache = {}
        for did in {pair[1] for pair in unjudged_pairs}:
            row = con.execute("SELECT record_json FROM documents WHERE doc_id = ?", (did,)).fetchone()
            doc_cache[did] = json.loads(row[0]) if row else {}
        con.close()

        queries_map = {q["query_id"]: q["query_text"] for q in queries}
        t0 = time.time()
        for idx, (qid, did) in enumerate(unjudged_pairs, 1):
            qtext = queries_map[qid]
            doc = doc_cache.get(did, {})
            title = doc.get("title", "")
            headings = " | ".join(doc.get("headings", [])[:6])
            body = doc.get("body_text", "")[:2500].replace("\n", " ").strip()
            code = doc.get("code_text", "")[:1200].replace("\n", " ").strip()

            prompt = f"""User Search Query: "{qtext}"

Candidate Document:
Title: {title}
Headings: {headings}
Explanatory Body: {body}
Code Snippets: {code}

Task:
Assign a graded relevance score:
- 2 (Highly Relevant): The document directly answers, explains, or resolves the user's specific query.
- 1 (Partially Relevant): The document covers the target technology or related subtopic, but does not completely address the query.
- 0 (Not Relevant): The document is off-topic, unrelated, or unhelpful.

Reply strictly in JSON: {{"grade": 2}}"""

            grade = call_gemma_grade(prompt)
            existing_graded[(qid, did)] = grade
            existing_binary[(qid, did)] = 1 if grade >= 1 else 0

            elapsed = time.time() - t0
            print(f"[{idx:02d}/{len(unjudged_pairs)}] ({elapsed:.1f}s) ({qid}, {did}): Grade {grade}", flush=True)

        # Save updated judgments
        updated_ensemble = []
        for (q, d), g in sorted(existing_graded.items()):
            updated_ensemble.append({
                "query_id": q,
                "doc_id": d,
                "consensus_grade": g,
                "lenient": 1 if g >= 1 else 0,
                "strict": 1 if g == 2 else 0,
            })
        with open(ENSEMBLE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "consensus_grade", "lenient", "strict"])
            writer.writeheader()
            writer.writerows(updated_ensemble)

        with open(JUDGMENTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "relevant", "updated_at"])
            writer.writeheader()
            for (q, d), g in sorted(existing_graded.items()):
                writer.writerow({
                    "query_id": q,
                    "doc_id": d,
                    "relevant": 1 if g >= 1 else 0,
                    "updated_at": "2026-11-05T20:00:00Z",
                })

    # Step 4: Calculate Comprehensive Evaluation Metrics
    print("\n=== [Step 4: Calculating Evaluation Metrics (Binary & Graded)] ===", flush=True)
    summary_results = []
    per_query_rows = []

    for (engine, variant), run_rows in all_runs.items():
        grouped_by_q = defaultdict(list)
        for r in run_rows:
            grouped_by_q[r["query_id"]].append(r)

        q_metrics = []
        for q in queries:
            qid = q["query_id"]
            results = sorted(grouped_by_q[qid], key=lambda x: int(x["rank"]))[:10]
            grades = [existing_graded.get((qid, r["doc_id"]), 0) for r in results]
            binaries = [existing_binary.get((qid, r["doc_id"]), 0) for r in results]

            p5 = sum(binaries[:5]) / 5.0
            p10 = sum(binaries[:10]) / 10.0
            # Total relevant in pool for this query
            pool_relevant = sum(existing_binary.get((qid, did), 0) for did in {pair[1] for pair in all_pool_pairs if pair[0] == qid})
            recall10 = sum(binaries[:10]) / pool_relevant if pool_relevant > 0 else 0.0
            rr10 = next((1.0 / rank for rank, b in enumerate(binaries, 1) if b == 1), 0.0)
            b_ndcg10 = compute_binary_ndcg(binaries, k=10)
            g_ndcg5 = compute_graded_ndcg(grades, k=5)
            g_ndcg10 = compute_graded_ndcg(grades, k=10)

            q_metrics.append({
                "p5": p5,
                "p10": p10,
                "recall10": recall10,
                "rr10": rr10,
                "b_ndcg10": b_ndcg10,
                "g_ndcg5": g_ndcg5,
                "g_ndcg10": g_ndcg10,
            })

            per_query_rows.append({
                "engine": engine,
                "variant": variant,
                "query_id": qid,
                "p5": f"{p5:.4f}",
                "p10": f"{p10:.4f}",
                "recall10": f"{recall10:.4f}",
                "rr10": f"{rr10:.4f}",
                "binary_ndcg10": f"{b_ndcg10:.4f}",
                "graded_ndcg5": f"{g_ndcg5:.4f}",
                "graded_ndcg10": f"{g_ndcg10:.4f}",
            })

        summary_results.append({
            "engine": engine,
            "variant": variant,
            "P@5": np.mean([m["p5"] for m in q_metrics]),
            "P@10": np.mean([m["p10"] for m in q_metrics]),
            "Recall@10": np.mean([m["recall10"] for m in q_metrics]),
            "MRR@10": np.mean([m["rr10"] for m in q_metrics]),
            "Binary_NDCG@10": np.mean([m["b_ndcg10"] for m in q_metrics]),
            "Graded_NDCG@5": np.mean([m["g_ndcg5"] for m in q_metrics]),
            "Graded_NDCG@10": np.mean([m["g_ndcg10"] for m in q_metrics]),
        })

    # Save per_query and metrics CSV
    with open(ROOT / "evaluation/results/per_query_graded.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "variant", "query_id", "p5", "p10", "recall10", "rr10", "binary_ndcg10", "graded_ndcg5", "graded_ndcg10"])
        writer.writeheader()
        writer.writerows(per_query_rows)

    with open(ROOT / "evaluation/results/metrics_graded.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "variant", "P@5", "P@10", "Recall@10", "MRR@10", "Binary_NDCG@10", "Graded_NDCG@5", "Graded_NDCG@10"])
        writer.writeheader()
        writer.writerows(summary_results)

    # Print Final Summary Table
    print("\n" + "=" * 115)
    print("FINAL BENCHMARK EVALUATION TABLE (ALL 15 QUERIES, GRADED + BINARY METRICS)")
    print("=" * 115)
    print(f"{'Method / Variant':26} | {'Graded NDCG@5':13} | {'Graded NDCG@10':14} | {'P@5':8} | {'P@10':8} | {'Recall@10':10} | {'MRR@10':8} | {'Binary NDCG@10':14}")
    print("-" * 115)
    for r in summary_results:
        name = f"{r['engine'].capitalize()} ({r['variant']})"
        print(
            f"{name:26} | {r['Graded_NDCG@5']:13.4f} | {r['Graded_NDCG@10']:14.4f} | "
            f"{r['P@5']:8.4f} | {r['P@10']:8.4f} | {r['Recall@10']:10.4f} | "
            f"{r['MRR@10']:8.4f} | {r['Binary_NDCG@10']:14.4f}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
