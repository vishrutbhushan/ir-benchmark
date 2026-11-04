#!/usr/bin/env python3
"""Run full evaluation pipeline: 4-variant retrieval, Ollama labeling, and metrics export."""

import argparse
import csv
import json
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "evaluation/queries/final_queries.jsonl"
JUDGMENTS_PATH = ROOT / "evaluation/data/judgments.csv"
DB_PATH = ROOT / "filter/outputs/stage_04_filtering/corpus.sqlite"
OLLAMA_URL = "http://172.23.64.1:11434/v1/chat/completions"
MODEL = "nemotron-3.5-bolt:e4b"

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


def run_phase_1_retrieval():
    """Retrieve top-10 for all 4 variants on the 10 evaluation queries."""
    print("=== [Phase 1: Retrieval Snapshots (4 Variants x 10 Queries)] ===", flush=True)
    from run_search_ui import SearchRuntime

    queries = [
        json.loads(line)
        for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(queries)} evaluation queries from {QUERIES_PATH}.", flush=True)

    runtime = SearchRuntime(
        ROOT / "neural/outputs",
        ROOT / "data/00_corpus",
        ROOT / "filter/outputs/stage_05_archive/technical_web_corpus.zip",
    )

    # 1. Classical: BM25 and BM25F
    print("\nLoading Classical Inverted Index...", flush=True)
    runtime._load_classical()
    for method, variant_name in [("bm25", "bm25"), ("bm25f", "bm25f")]:
        rows = []
        t0 = time.perf_counter()
        for q in queries:
            qid = q["query_id"]
            qtext = q["query_text"]
            res = runtime.search(method, qtext)
            for r in res["results"][:10]:
                rows.append({
                    "engine": "classical",
                    "variant": variant_name,
                    "query_id": qid,
                    "query_text": qtext,
                    "rank": r["rank"],
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "url": r["url"],
                    "technology": r["technology"],
                    "categories": r["category"],
                    "score": r["score"],
                })
        elapsed = (time.perf_counter() - t0) * 1000
        out_path = ROOT / f"evaluation/results/classical/{variant_name}.csv"
        write_csv(out_path, rows)
        print(f"  [Classical] {variant_name}: exported {len(rows)} results to {out_path} ({elapsed:.1f}ms total)", flush=True)

    # 2. Neural: Dense and Zones (Routed Cluster)
    print("\nLoading Neural Search Engine & GPU embeddings...", flush=True)
    runtime._load_neural()
    for mode, variant_name in [("dense", "dense"), ("routed-cluster", "zones")]:
        rows = []
        t0 = time.perf_counter()
        for q in queries:
            qid = q["query_id"]
            qtext = q["query_text"]
            raw_res = runtime.neural.search(qtext, mode=mode, page_size=10)["results"]
            for rank, r in enumerate(raw_res[:10], 1):
                rows.append({
                    "engine": "neural",
                    "variant": variant_name,
                    "query_id": qid,
                    "query_text": qtext,
                    "rank": rank,
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "url": r["url"],
                    "technology": r.get("technology", ""),
                    "categories": r.get("category", r.get("categories", "")),
                    "score": round(float(r["score"]), 6),
                })
        elapsed = (time.perf_counter() - t0) * 1000
        out_path = ROOT / f"evaluation/results/neural/{variant_name}.csv"
        write_csv(out_path, rows)
        print(f"  [Neural] {variant_name} ({mode}): exported {len(rows)} results to {out_path} ({elapsed:.1f}ms total)", flush=True)

    print("\nPhase 1 complete. All retrieval snapshots generated.\n", flush=True)


def call_ollama(prompt, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "format": "json",
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    msg = data["choices"][0]["message"]
    return msg.get("content", "").strip()


def run_phase_2_labeling():
    """Pool candidates across all runs and evaluate with local Ollama."""
    print("=== [Phase 2: Local LLM Pool Labeling] ===", flush=True)
    # Collect all unique (qid, qtext, doc_id) across all runs
    pool = {}
    for engine in ("classical", "neural"):
        for path in (ROOT / f"evaluation/results/{engine}").glob("*.csv"):
            with path.open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    pool[(r["query_id"], r["doc_id"])] = {
                        "query_id": r["query_id"],
                        "query_text": r["query_text"],
                        "doc_id": r["doc_id"],
                    }

    print(f"Pooled candidate pairs across all variants: {len(pool)} unique pairs.", flush=True)

    # Read existing judgments
    existing = {}
    if JUDGMENTS_PATH.exists():
        with JUDGMENTS_PATH.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing[(r["query_id"], r["doc_id"])] = r
    else:
        with JUDGMENTS_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "doc_id", "relevant", "updated_at"])

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    items = list(pool.values())
    total = len(items)
    newly_judged = 0

    for idx, item in enumerate(items, 1):
        qid = item["query_id"]
        doc_id = item["doc_id"]
        qtext = item["query_text"]

        if (qid, doc_id) in existing:
            continue

        row = cur.execute(
            "SELECT record_json FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            continue
        doc = json.loads(row[0])

        title = doc.get("title", "")
        headings = " | ".join(doc.get("headings", [])[:6])
        body_snippet = doc.get("body_text", "")[:800].replace("\n", " ")
        code_snippet = doc.get("code_text", "")[:800].replace("\n", " ")

        prompt = f"""You are an expert technical Information Retrieval judge.
Evaluate whether the following technical document is RELEVANT or NOT_RELEVANT to the search query.
A document is RELEVANT (1) if it directly answers, explains, documents, or provides solutions/code for the query.
Otherwise, it is NOT_RELEVANT (0).

Search Query: "{qtext}"

Candidate Document:
- Title: {title}
- Headings: {headings}
- Body Excerpt: {body_snippet}
- Code Excerpt: {code_snippet}

Respond ONLY in valid JSON format:
{{"relevant": 1 or 0, "rationale": "one sentence explanation"}}"""

        try:
            raw_resp = call_ollama(prompt)
            start = raw_resp.find("{")
            end = raw_resp.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw_resp[start:end])
            else:
                raise ValueError(f"Could not parse JSON: {raw_resp}")

            relevant = int(parsed.get("relevant", 0))
            rationale = str(parsed.get("rationale", "")).replace("\n", " ")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Append incrementally to disk immediately
            with JUDGMENTS_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([qid, doc_id, relevant, now])

            existing[(qid, doc_id)] = {"relevant": relevant}
            newly_judged += 1
            status_tag = "RELEVANT (1)" if relevant == 1 else "NOT_RELEVANT (0)"
            print(
                f"[{idx}/{total}] Judged {qid} vs {doc_id}: {status_tag} | {rationale[:65]}...",
                flush=True,
            )

        except Exception as err:
            print(f"[{idx}/{total}] Error judging ({qid}, {doc_id}): {err}", flush=True)

    con.close()
    print(f"\nPhase 2 complete. Newly judged: {newly_judged}. Total judgments in store: {len(existing)} pairs.\n", flush=True)


def run_phase_3_metrics():
    """Compute and display IR evaluation metrics."""
    print("=== [Phase 3: Metrics Calculation & Export] ===", flush=True)
    import subprocess
    cmd = [str(ROOT / ".venv/bin/python"), "evaluation/run_evaluation.py", "--metrics", "--force"]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    metrics_csv = ROOT / "evaluation/results/metrics.csv"
    if metrics_csv.exists():
        print("\n--- Final Comparative IR Evaluation Metrics ---")
        with metrics_csv.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        header = f"{'Engine':<12} | {'Variant':<10} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'P@5':<6} | {'P@10':<6} | {'Recall@10':<10} | {'MAP@10':<8} | {'MRR@10':<8}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r.get('engine', ''):<12} | {r.get('variant', ''):<10} | "
                f"{float(r.get('pooled_ndcg@5', 0)):<8.4f} | {float(r.get('pooled_ndcg@10', 0)):<8.4f} | "
                f"{float(r.get('precision@5', 0)):<6.4f} | {float(r.get('precision@10', 0)):<6.4f} | "
                f"{float(r.get('pooled_recall@10', 0)):<10.4f} | {float(r.get('pooled_ap@10', 0)):<8.4f} | "
                f"{float(r.get('rr@10', 0)):<8.4f}"
            )
        print("-" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["retrieval", "label", "metrics", "all"], default="all")
    args = parser.parse_args()

    if args.step in {"retrieval", "all"}:
        run_phase_1_retrieval()
    if args.step in {"label", "all"}:
        run_phase_2_labeling()
    if args.step in {"metrics", "all"}:
        run_phase_3_metrics()

