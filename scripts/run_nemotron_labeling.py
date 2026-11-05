#!/usr/bin/env python3
"""Run Judge 1 (nemotron-3.5-bolt:e4b) labeling on unjudged candidate pairs."""

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
OLLAMA_URL = "http://172.23.64.1:11434/v1/chat/completions"
MODEL = "nemotron-3.5-bolt:e4b"


def call_nemotron(prompt, max_tokens=1024):
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


def run_nemotron_evaluation(workers=2):
    print("=== [Running nemotron-3.5-bolt:e4b Labeling (Precision / Strict)] ===", flush=True)

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

    print(f"Total candidate pairs in pool: {len(pool)}", flush=True)

    existing = set()
    if NEMOTRON_CSV.exists():
        with NEMOTRON_CSV.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("relevant") in ("0", "1"):
                    existing.add((r["query_id"], r["doc_id"]))
    else:
        with NEMOTRON_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "doc_id", "relevant", "updated_at"])

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

    if not candidates_to_process:
        print("All candidate pairs are already judged by Nemotron.\n")
        return

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

        for attempt in range(2):
            try:
                raw_resp = call_nemotron(prompt)
                start = raw_resp.find("{")
                end = raw_resp.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw_resp[start:end])
                else:
                    raise ValueError(f"Could not parse JSON: {raw_resp}")

                rel = int(parsed.get("relevant", 0))
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                with csv_lock:
                    with NEMOTRON_CSV.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([qid, doc_id, rel, now])
                    completed_count += 1
                    status_tag = "RELEVANT (1)" if rel == 1 else "NOT_RELEVANT (0)"
                    print(
                        f"[{completed_count}/{total_count}] [Nemotron] {qid} vs {doc_id}: {status_tag} | {title[:60]}",
                        flush=True,
                    )
                return True
            except Exception as err:
                if attempt == 1:
                    print(f"Error judging ({qid}, {doc_id}): {err}", flush=True)
                    return False
                time.sleep(1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(judge_pair, item) for item in candidates_to_process]
        for _ in as_completed(futures):
            pass

    print(f"\nNemotron judging complete. All judgments saved to {NEMOTRON_CSV}.\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_nemotron_evaluation(workers=args.workers)

