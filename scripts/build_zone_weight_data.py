#!/usr/bin/env python3
"""Build zone weight training dataset using Phase 1 (retrieval) and Phase 2 (Ollama labeling)."""

import argparse
import csv
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "filter/outputs/stage_04_filtering/corpus.sqlite"
QUERIES_PATH = ROOT / "data_inputs/neural_weights/zone_weight_queries.csv"
CANDIDATES_PATH = ROOT / "data_inputs/neural_weights/candidates.json"
JUDGMENTS_PATH = ROOT / "data_inputs/neural_weights/zone_weight_judgments.csv"
OLLAMA_URL = "http://172.23.64.1:11434/v1/chat/completions"
MODEL = "nemotron-3.5-bolt:e4b"


def run_phase_1():
    """Retrieve candidate documents using BM25, then clean up memory."""
    print("--- [Phase 1: Candidate Retrieval] ---", flush=True)
    with QUERIES_PATH.open(encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    # Classical BM25
    sys.path.insert(0, str(ROOT / "classical"))
    from src.retrieval.bm25_scorer import BM25Scorer
    from src.retrieval.inverted_index import InvertedIndex
    from src.retrieval.query_processor import QueryProcessor

    index_path = ROOT / "classical/outputs/stage_03_retrieval/inverted_index.json"
    print(f"Loading inverted index from {index_path}...", flush=True)
    index = InvertedIndex.load(index_path)
    processor = QueryProcessor(index, BM25Scorer(index))

    # Also pick a couple of diverse background documents to guarantee clear negative examples
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    random_docs = [
        row[0]
        for row in cur.execute(
            "SELECT doc_id FROM documents WHERE technology IN ('Kubernetes', 'Rust', 'PostgreSQL', 'Docker', 'Python') ORDER BY doc_index DESC LIMIT 20"
        ).fetchall()
    ]
    con.close()

    candidates = []
    seen = set()

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        print(f"Searching candidates for {qid}: '{qtext}'...", flush=True)
        results = processor.search(qtext, top_k=4)
        for r in results.get("results", []):
            pair = (qid, r["doc_id"])
            if pair not in seen:
                seen.add(pair)
                candidates.append({"query_id": qid, "query_text": qtext, "doc_id": r["doc_id"]})
        # Add 1-2 distinct documents from other technologies as known negative candidates
        for neg_id in random_docs:
            pair = (qid, neg_id)
            if pair not in seen and len([c for c in candidates if c["query_id"] == qid]) < 6:
                seen.add(pair)
                candidates.append({"query_id": qid, "query_text": qtext, "doc_id": neg_id})

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    print(f"Phase 1 complete. Saved {len(candidates)} candidates to {CANDIDATES_PATH}.\n", flush=True)


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


def run_phase_2():
    """Call Ollama to annotate candidates, saving incrementally."""
    print("--- [Phase 2: Ollama Annotation] ---", flush=True)
    if not CANDIDATES_PATH.exists():
        raise RuntimeError("Candidates file not found. Run phase 1 first.")

    with CANDIDATES_PATH.open(encoding="utf-8") as f:
        candidates = json.load(f)

    # Load existing judgments to allow resuming seamlessly
    existing = set()
    if JUDGMENTS_PATH.exists():
        with JUDGMENTS_PATH.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["query_id"], row["doc_id"]))
    else:
        with JUDGMENTS_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id",
                "doc_id",
                "relevance_grade",
                "evidence_zone",
                "evidence_text",
                "rationale",
            ])

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    total = len(candidates)
    processed = 0

    for idx, item in enumerate(candidates, 1):
        qid = item["query_id"]
        doc_id = item["doc_id"]
        qtext = item["query_text"]

        if (qid, doc_id) in existing:
            print(f"[{idx}/{total}] Skipping already judged: ({qid}, {doc_id})", flush=True)
            continue

        row = cur.execute(
            "SELECT record_json FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            continue
        doc = json.loads(row[0])

        title = doc.get("title", "")
        headings = " | ".join(doc.get("headings", [])[:6])
        body_snippet = doc.get("body_text", "")[:700].replace("\n", " ")
        code_snippet = doc.get("code_text", "")[:700].replace("\n", " ")

        prompt = f"""You are an Information Retrieval judge.
Evaluate if this technical document is relevant to the search query.

Query: "{qtext}"

Document Info:
- Title: {title}
- Headings: {headings}
- Body Excerpt: {body_snippet}
- Code Excerpt: {code_snippet}

Instructions:
1. "relevant": 1.0 if it answers or contains relevant information for the query, else 0.0.
2. "evidence_zone": Which zone contains the primary evidence? Must be one of: "title", "headings", "body", "code".
3. "evidence_text": A short snippet (under 100 characters) showing the evidence.
4. "rationale": One sentence explaining the decision.

Respond ONLY with valid JSON:
{{"relevance_grade": 1.0 or 0.0, "evidence_zone": "title"|"headings"|"body"|"code", "evidence_text": "...", "rationale": "..."}}"""

        try:
            raw_resp = call_ollama(prompt)
            # Find the JSON substring if any markdown wrappers exist
            start = raw_resp.find("{")
            end = raw_resp.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw_resp[start:end])
            else:
                raise ValueError(f"Could not find JSON in response: {raw_resp}")

            grade = float(parsed.get("relevance_grade", 0.0))
            zone = str(parsed.get("evidence_zone", "body")).lower()
            if zone not in {"title", "headings", "body", "code"}:
                zone = "body"
            evidence = str(parsed.get("evidence_text", ""))[:120].replace("\n", " ")
            rationale = str(parsed.get("rationale", ""))[:200].replace("\n", " ")

            # Append incrementally to disk immediately
            with JUDGMENTS_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([qid, doc_id, f"{grade:.2f}", zone, evidence, rationale])

            existing.add((qid, doc_id))
            processed += 1
            print(
                f"[{idx}/{total}] Judged {qid} vs {doc_id}: grade={grade}, zone={zone} | {rationale[:60]}...",
                flush=True,
            )

        except Exception as err:
            print(f"[{idx}/{total}] Error judging ({qid}, {doc_id}): {err}", flush=True)

    con.close()
    print(f"\nPhase 2 complete. Total newly judged: {processed}. Current dataset size: {len(existing)} pairs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    if args.phase in {"1", "all"}:
        run_phase_1()
    if args.phase in {"2", "all"}:
        run_phase_2()

