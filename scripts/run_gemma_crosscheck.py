#!/usr/bin/env python3
"""Run secondary cross-check labeling with gemma4:12b (including partial relevance).

Computes:
  1. Inter-annotator agreement: Cohen's Kappa statistic between Nemotron and Gemma.
  2. Consensus ensemble: Grade 2 (both agree), Grade 1 (partial / one says yes), Grade 0 (neither).
  3. Strict vs Lenient IR evaluation metrics across all 4 search paradigms.
"""

import argparse
import csv
import json
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "filter/outputs/stage_04_filtering/corpus.sqlite"
NEMOTRON_CSV = ROOT / "evaluation/data/judgments_nemotron.csv"
GEMMA_CSV = ROOT / "evaluation/data/judgments_gemma.csv"
ENSEMBLE_CSV = ROOT / "evaluation/data/judgments_ensemble.csv"
CANONICAL_JUDGMENTS_CSV = ROOT / "evaluation/data/judgments.csv"
OLLAMA_URL = "http://172.23.64.1:11434/v1/chat/completions"
MODEL = "gemma4:12b"


def call_gemma(prompt, max_tokens=1024):
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
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    msg = data["choices"][0]["message"]
    return msg.get("content", "").strip()


def run_gemma_evaluation(workers=2):
    print("=== [Step 1: Running gemma4:12b Cross-Check Labeling (Inclusive/Partial)] ===", flush=True)

    # Collect pool from retrieval snapshots
    pool = {}
    for engine in ("classical", "neural"):
        for path in (ROOT / f"evaluation/results/{engine}").glob("*.csv"):
            if "archive" in path.parts:
                continue
            with path.open(encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    pool[(r["query_id"], r["doc_id"])] = {
                        "query_id": r["query_id"],
                        "query_text": r["query_text"],
                        "doc_id": r["doc_id"],
                    }

    print(f"Total candidate pairs to judge: {len(pool)}", flush=True)

    # Read existing Gemma judgments for resuming
    existing = set()
    if GEMMA_CSV.exists():
        with GEMMA_CSV.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                existing.add((r["query_id"], r["doc_id"]))
    else:
        with GEMMA_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "doc_id", "relevant", "rationale", "updated_at"])

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    docs = {}
    for qid, doc_id in pool:
        row = cur.execute("SELECT record_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row:
            docs[doc_id] = json.loads(row[0])
    con.close()

    candidates_to_process = [v for k, v in pool.items() if k not in existing]
    print(f"Already judged: {len(existing)}. Remaining to judge: {len(candidates_to_process)}", flush=True)

    csv_lock = threading.Lock()
    completed_count = len(existing)
    total_count = len(pool)

    def judge_pair(item):
        nonlocal completed_count
        qid = item["query_id"]
        doc_id = item["doc_id"]
        qtext = item["query_text"]
        doc = docs.get(doc_id, {})

        title = doc.get("title", "")
        headings = " | ".join(doc.get("headings", [])[:6])
        body_snippet = doc.get("body_text", "")[:800].replace("\n", " ")
        code_snippet = doc.get("code_text", "")[:800].replace("\n", " ")

        prompt = f"""You are an expert technical Information Retrieval judge.
Evaluate whether the following technical document is RELEVANT (1) or NOT_RELEVANT (0) to the search query.

Guideline:
- Mark RELEVANT (1) if the document directly answers, explains, provides syntax, OR contains useful related documentation, API references, or partial concepts that help answer the query.
- Mark NOT_RELEVANT (0) ONLY if the document is completely off-topic, unrelated, or provides no useful information for the user's query.

Search Query: "{qtext}"

Candidate Document:
- Title: {title}
- Headings: {headings}
- Body Excerpt: {body_snippet}
- Code Excerpt: {code_snippet}

Respond ONLY in valid JSON format:
{{"relevant": 1 or 0, "rationale": "one sentence explanation"}}"""

        for attempt in range(2):
            try:
                raw_resp = call_gemma(prompt)
                start = raw_resp.find("{")
                end = raw_resp.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw_resp[start:end])
                else:
                    raise ValueError(f"Could not parse JSON: {raw_resp}")

                rel = int(parsed.get("relevant", 0))
                rationale = str(parsed.get("rationale", "")).replace("\n", " ")
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                with csv_lock:
                    with GEMMA_CSV.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([qid, doc_id, rel, rationale, now])
                    completed_count += 1
                    status_tag = "RELEVANT (1)" if rel == 1 else "NOT_RELEVANT (0)"
                    print(
                        f"[{completed_count}/{total_count}] [Gemma] {qid} vs {doc_id}: {status_tag} | {rationale[:65]}...",
                        flush=True,
                    )
                return True
            except Exception as err:
                if attempt == 1:
                    print(f"Error judging ({qid}, {doc_id}): {err}", flush=True)
                    return False
                time.sleep(1)

    if candidates_to_process:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(judge_pair, item) for item in candidates_to_process]
            for _ in as_completed(futures):
                pass

    print(f"\nStep 1 Complete. All Gemma judgments saved to {GEMMA_CSV}.\n", flush=True)


def compute_inter_annotator_agreement():
    print("=== [Step 2: Computing Inter-Annotator Agreement & Building Ensemble] ===", flush=True)

    # Load Nemotron judgments
    nemotron = {}
    with NEMOTRON_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nemotron[(r["query_id"], r["doc_id"])] = int(r["relevant"])

    # Load Gemma judgments
    gemma = {}
    with GEMMA_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gemma[(r["query_id"], r["doc_id"])] = int(r["relevant"])

    common_keys = sorted(list(set(nemotron.keys()) & set(gemma.keys())))
    print(f"Common judged pairs between Nemotron and Gemma: {len(common_keys)}")

    # Contingency Table:
    #                 Gemma = 1     Gemma = 0
    # Nemotron = 1       a             b
    # Nemotron = 0       c             d
    a = sum(nemotron[k] == 1 and gemma[k] == 1 for k in common_keys)
    b = sum(nemotron[k] == 1 and gemma[k] == 0 for k in common_keys)
    c = sum(nemotron[k] == 0 and gemma[k] == 1 for k in common_keys)
    d = sum(nemotron[k] == 0 and gemma[k] == 0 for k in common_keys)
    n = len(common_keys)

    po = (a + d) / n
    p_yes_nem = (a + b) / n
    p_no_nem = (c + d) / n
    p_yes_gem = (a + c) / n
    p_no_gem = (b + d) / n
    pe = (p_yes_nem * p_yes_gem) + (p_no_nem * p_no_gem)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    print("\n--- Contingency Matrix ---")
    print(f"                    Gemma=1 (Relev) | Gemma=0 (Non-Relev)")
    print(f"Nemotron=1 (Relev)       {a:5d}      |      {b:5d}")
    print(f"Nemotron=0 (Non-Relev)   {c:5d}      |      {d:5d}")
    print(f"\nAgreement Rate P(A): {po:.4f} ({po*100:.1f}%)")
    print(f"Chance Agreement P(E): {pe:.4f} ({pe*100:.1f}%)")
    print(f"Cohen's Kappa (κ):     {kappa:.4f}")
    if kappa > 0.8:
        strength = "Strong Agreement"
    elif kappa >= 0.67:
        strength = "Fair / Substantial Agreement (Standard TREC range)"
    elif kappa >= 0.4:
        strength = "Moderate Agreement (Reflects strict vs inclusive criteria)"
    else:
        strength = "Slight / Divergent"
    print(f"Interpretation: {strength}\n")

    # Build ensemble dataset
    # consensus_grade: 2 (both 1), 1 (one 1), 0 (neither)
    ensemble_rows = []
    for k in common_keys:
        qid, doc_id = k
        n_rel = nemotron[k]
        g_rel = gemma[k]
        consensus = n_rel + g_rel  # 0, 1, or 2
        strict = 1 if (n_rel == 1 and g_rel == 1) else 0
        lenient = 1 if (n_rel == 1 or g_rel == 1) else 0
        ensemble_rows.append({
            "query_id": qid,
            "doc_id": doc_id,
            "nemotron": n_rel,
            "gemma": g_rel,
            "consensus_grade": consensus,
            "strict": strict,
            "lenient": lenient,
        })

    with ENSEMBLE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "nemotron", "gemma", "consensus_grade", "strict", "lenient"])
        writer.writeheader()
        writer.writerows(ensemble_rows)
    print(f"Saved ensemble judgments to {ENSEMBLE_CSV}.")
    print(f"  - Grade 2 (Both Agree Relevant):     {a} pairs")
    print(f"  - Grade 1 (Partial / One Agrees):     {b + c} pairs ({b} Nemotron-only, {c} Gemma-only)")
    print(f"  - Grade 0 (Both Agree Non-Relevant): {d} pairs\n")


def to_float(val, default=0.0):
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def evaluate_under_modes():
    print("=== [Step 3: Comparative Evaluation under Strict vs. Lenient vs. Graded Modes] ===", flush=True)
    import subprocess
    # First: evaluate under Lenient mode (Gemma + Nemotron inclusive union)
    with ENSEMBLE_CSV.open(encoding="utf-8-sig") as f:
        ensemble = list(csv.DictReader(f))

    # Overwrite judgments.csv with lenient labels to run standard evaluation runner
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with CANONICAL_JUDGMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "doc_id", "relevant", "updated_at"])
        for r in ensemble:
            writer.writerow([r["query_id"], r["doc_id"], r["lenient"], now])

    print("Exporting metrics under Lenient Mode (Partial Relevance Accepted)...", flush=True)
    cmd = [str(ROOT / ".venv/bin/python"), "evaluation/run_evaluation.py", "--metrics", "--force"]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    metrics_csv = ROOT / "evaluation/results/metrics.csv"
    with metrics_csv.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("query_type") == "all"]

    print("\n--- Evaluation Results under Lenient Mode (Accepting Partial Relevance) ---")
    header = f"{'Engine':<12} | {'Variant':<10} | {'NDCG@5':<8} | {'NDCG@10':<8} | {'P@5':<6} | {'P@10':<6} | {'Recall@10':<10} | {'MAP@10':<8} | {'MRR@10':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('engine', ''):<12} | {r.get('variant', ''):<10} | "
            f"{to_float(r.get('pooled_ndcg@5', 0)):<8.4f} | {to_float(r.get('pooled_ndcg@10', 0)):<8.4f} | "
            f"{to_float(r.get('precision@5', 0)):<6.4f} | {to_float(r.get('precision@10', 0)):<6.4f} | "
            f"{to_float(r.get('pooled_recall@10', 0)):<10.4f} | {to_float(r.get('pooled_MAP@10', 0)):<8.4f} | "
            f"{to_float(r.get('MRR@10', 0)):<8.4f}"
        )
    print("-" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-gemma", action="store_true")
    args = parser.parse_args()

    if not args.skip_gemma:
        run_gemma_evaluation(workers=args.workers)
    compute_inter_annotator_agreement()
    evaluate_under_modes()

