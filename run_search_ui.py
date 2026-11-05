#!/usr/bin/env python3
"""Serve a fast, read-only search UI evaluating BM25, BM25F, Neural, and Hybrid RRF."""

import argparse
import html
import json
import logging
import os
import re
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent

# Auto-switch to virtual environment if current interpreter lacks PyTorch
try:
    import torch
except ImportError:
    venv_py = ROOT / ".venv/bin/python"
    if venv_py.exists() and sys.executable != str(venv_py):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)


def get_configured_limit():
    """Read the configured result limit from config/pipeline.json (default 10, adaptable to 20)."""
    try:
        cfg_path = ROOT / "config/pipeline.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return int(cfg.get("neural", {}).get("result_limit", cfg.get("classical", {}).get("top_k", 10)))
    except Exception:
        pass
    return 10


def clean_title_for_comparison(title):
    return re.sub(r"\s*[-|–—:]\s*.*$", "", title or "").strip().lower()


def extract_heading_snippet(title, headings):
    """Return a breadcrumb-style snippet of distinct headings.
    
    If a heading is identical to the title we are already displaying, it is omitted.
    """
    if not headings:
        return ""
    norm_title = clean_title_for_comparison(title)
    distinct = []
    seen = set()
    for h in headings:
        h_clean = re.sub(r"[\s#]+$", "", str(h or "")).strip()
        if not h_clean:
            continue
        h_norm = h_clean.lower()
        if h_norm == norm_title or (len(h_norm) > 3 and h_norm in norm_title):
            continue
        if h_norm not in seen:
            seen.add(h_norm)
            distinct.append(h_clean)
        if len(distinct) >= 3:
            break
    if not distinct and headings:
        h_first = re.sub(r"[\s#]+$", "", str(headings[0])).strip()
        if h_first.lower() != norm_title:
            distinct = [h_first]
    return " › ".join(distinct)


METHOD_NAMES = {
    "hybrid-rrf": "Hybrid RRF (BM25F + Routed Neural)",
    "bm25f": "Hierarchical BM25F (Fields)",
    "bm25": "Flat BM25 (Lucene Default)",
    "neural-routed": "Cluster-Routed Neural (Soft Boundary)",
    "neural-dense": "Dense Cosine (Exhaustive)",
}


class SearchRuntime:
    def __init__(self, neural_output, corpus_dir, corpus_zip):
        self.classical = {}
        self.neural = None
        self.neural_output = neural_output
        self.corpus_dir = corpus_dir
        self.corpus_zip = corpus_zip
        self.sqlite_con = None
        self._init_sqlite()

    def _init_sqlite(self):
        db_path = self.neural_output / "step_01_extraction/corpus.sqlite"
        if not db_path.exists():
            db_path = ROOT / "filter/outputs/stage_04_filtering/corpus.sqlite"
        if db_path.exists():
            try:
                uri = f"file:{db_path.resolve()}?mode=ro"
                self.sqlite_con = sqlite3.connect(uri, uri=True, check_same_thread=False)
            except Exception:
                self.sqlite_con = None

    def get_document_headings(self, doc_id, default_title=""):
        if not self.sqlite_con:
            return ""
        try:
            row = self.sqlite_con.execute(
                "SELECT title, record_json FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row:
                title = row[0] or default_title
                data = json.loads(row[1]) if row[1] else {}
                headings = data.get("headings", [])
                return extract_heading_snippet(title, headings)
        except Exception:
            pass
        return ""

    def _load_classical(self):
        if self.classical:
            return
        classical = ROOT / "classical"
        saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        for k in list(saved_src.keys()):
            del sys.modules[k]
        if str(classical) not in sys.path:
            sys.path.insert(0, str(classical))
        try:
            from src.retrieval.bm25_scorer import BM25Scorer
            from src.retrieval.bm25f_scorer import BM25FScorer
            from src.retrieval.inverted_index import InvertedIndex
            from src.retrieval.query_processor import QueryProcessor

            index_path = classical / "outputs/stage_03_retrieval/inverted_index.json"
            if not index_path.exists():
                raise RuntimeError("Classical index is unavailable; run classical/run_pipeline.py first")
            index = InvertedIndex.load(index_path)
            self.classical = {
                "bm25": QueryProcessor(index, BM25Scorer(index)),
                "bm25f": QueryProcessor(index, BM25FScorer(index)),
            }
        finally:
            if str(classical) in sys.path:
                sys.path.remove(str(classical))
            for k in list(sys.modules.keys()):
                if k == "src" or k.startswith("src."):
                    del sys.modules[k]
            sys.modules.update(saved_src)

    def _load_neural(self):
        if self.neural is not None:
            return
        neural = ROOT / "neural"
        saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        for k in list(saved_src.keys()):
            del sys.modules[k]
        if str(neural) not in sys.path:
            sys.path.insert(0, str(neural))
        try:
            os.environ.setdefault("NEURAL_CONFIG", str(ROOT / "config/pipeline.json"))
            try:
                import torch
            except ImportError:
                print("  [!] PyTorch not found. Neural search is disabled; Classical BM25/BM25F search is active.", flush=True)
                self.neural = False
                return
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from src.retrieval.step_08_search import SearchEngine

            self.neural = SearchEngine(
                self.neural_output,
                device,
                self.corpus_dir,
                corpus_zip=self.corpus_zip,
            )
        finally:
            pass

    def search(self, method, query, limit=None):
        limit = limit or get_configured_limit()
        started = time.perf_counter()

        if method == "hybrid-rrf":
            self._load_classical()
            self._load_neural()
            fetch_k = limit * 2
            bm25f_rows = self.classical["bm25f"].search(query, top_k=fetch_k)["results"]
            if self.neural:
                routed_rows = self.neural.search(query, mode="routed-cluster", page_size=fetch_k)["results"]
            else:
                routed_rows = []

            # Reciprocal Rank Fusion: 0.5 / (60 + r_bm25f) + 0.5 / (60 + r_routed)
            scores = {}
            doc_map = {}
            for rank, r in enumerate(bm25f_rows, 1):
                did = r["doc_id"]
                scores[did] = scores.get(did, 0.0) + 0.5 / (60.0 + rank)
                doc_map[did] = r
            for rank, r in enumerate(routed_rows, 1):
                did = r["doc_id"]
                scores[did] = scores.get(did, 0.0) + 0.5 / (60.0 + rank)
                if did not in doc_map:
                    doc_map[did] = r

            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            rows = []
            for did, fused_score in sorted_docs:
                item = dict(doc_map[did])
                item["score"] = fused_score
                rows.append(item)

        elif method in {"bm25", "bm25f"}:
            self._load_classical()
            payload = self.classical[method].search(query, top_k=limit)
            rows = payload["results"]
        else:
            self._load_neural()
            if not self.neural:
                raise RuntimeError("Neural search requires PyTorch. Run with '.venv/bin/python run_search_ui.py' or activate .venv.")
            mode = "routed-cluster" if method == "neural-routed" else "dense"
            rows = self.neural.search(query, mode=mode, page_size=limit)["results"]

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        results = [self._row(row, rank) for rank, row in enumerate(rows, 1)]

        return {
            "method": method,
            "method_name": METHOD_NAMES.get(method, method),
            "query": query,
            "limit": limit,
            "elapsed_ms": elapsed_ms,
            "results": results,
        }

    def _row(self, row, rank):
        title = row.get("title", "")
        doc_id = row.get("doc_id", "")
        heading_snippet = self.get_document_headings(doc_id, title)

        return {
            "rank": rank,
            "doc_id": doc_id,
            "title": title,
            "technology": row.get("technology", ""),
            "heading_snippet": heading_snippet,
            "score": round(float(row.get("score", 0)), 6),
            "excerpt": row.get("snippet", row.get("body_text", ""))[:350],
            "url": row.get("url", ""),
        }


def serve(host, port, neural_output, corpus_dir, corpus_zip):
    print("Preloading search engines into memory...", flush=True)
    runtime = SearchRuntime(neural_output, corpus_dir, corpus_zip)
    print("  [1/2] Loading Classical inverted index into memory...", flush=True)
    runtime._load_classical()
    print("  [2/2] Loading Neural search engine & model weights into memory...", flush=True)
    runtime._load_neural()
    if runtime.neural:
        print("  Warming up search caches...", flush=True)
        runtime.neural.warmup()
    print("Ready! All available search engines are resident in memory.\n", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[search-ui] {fmt % args}", flush=True)

        def respond(self, payload, status=200, content_type="application/json"):
            try:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self.respond(UI.encode(), content_type="text/html")
                return
            if parsed.path == "/favicon.ico":
                try:
                    self.send_response(204)
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if parsed.path == "/api/search":
                values = parse_qs(parsed.query)
                query = values.get("q", [""])[0].strip()
                method = values.get("method", ["hybrid-rrf"])[0]
                limit_arg = values.get("limit", [None])[0]
                limit = int(limit_arg) if limit_arg and limit_arg.isdigit() else None
                if not query or method not in METHOD_NAMES:
                    self.respond({"error": "Provide query 'q' and a supported retrieval method"}, 400)
                    return
                try:
                    self.respond(runtime.search(method, query, limit=limit))
                except (RuntimeError, ValueError, FileNotFoundError) as error:
                    logging.exception("Search request failed")
                    self.respond({"error": str(error)}, 503)
                return
            self.send_error(404)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Search UI running at: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if runtime.neural:
            runtime.neural.close()
        if runtime.sqlite_con:
            runtime.sqlite_con.close()
        server.server_close()


UI = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IR Benchmark: Technical Documentation Search</title>
<style>
  :root {
    --primary: #1a73e8;
    --text: #202124;
    --subtext: #5f6368;
    --border: #dadce0;
    --bg-light: #f8f9fa;
    --chip-bg: #e8f0fe;
    --chip-text: #1967d2;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text);
    background: #fff;
    max-width: 960px;
    margin: 30px auto;
    padding: 0 20px;
    line-height: 1.5;
  }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 20px; color: #17194a; }
  .search-form { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .input-query {
    flex: 1;
    min-width: 280px;
    padding: 12px 16px;
    font-size: 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    outline: none;
    transition: border-color 0.15s;
  }
  .input-query:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(26,115,232,0.2); }
  .select-method {
    padding: 12px 14px;
    font-size: 15px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #fff;
    cursor: pointer;
  }
  .btn-search {
    padding: 12px 24px;
    font-size: 15px;
    font-weight: 600;
    color: #fff;
    background: var(--primary);
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
  }
  .btn-search:hover { background: #1557b0; }
  .status-chip {
    display: inline-block;
    padding: 6px 14px;
    font-size: 13.5px;
    background: var(--chip-bg);
    color: var(--chip-text);
    border-radius: 16px;
    margin-bottom: 20px;
    font-weight: 500;
  }
  .status-chip.error { background: #fce8e6; color: #c5221f; }
  .results-container { display: flex; flex-direction: column; gap: 20px; }
  .result-card {
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
  }
  .result-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 4px;
  }
  .rank-badge {
    font-size: 13px;
    font-weight: 700;
    color: var(--subtext);
  }
  .result-title {
    font-size: 18px;
    font-weight: 500;
    color: #1a0dab;
    text-decoration: none;
    line-height: 1.35;
  }
  .result-title:hover { text-decoration: underline; }
  .result-url-bar {
    font-size: 12.5px;
    color: #202124;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    word-break: break-all;
  }
  .tech-badge {
    background: #f1f3f4;
    color: #3c4043;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .url-text { color: #006621; }
  .heading-snippet {
    font-size: 13px;
    color: #4d5156;
    background: #f8f9fa;
    border-left: 3px solid #1a73e8;
    padding: 4px 10px;
    border-radius: 0 4px 4px 0;
    margin-bottom: 8px;
    font-weight: 500;
  }
  .heading-snippet .crumb-icon { color: #1a73e8; font-weight: bold; margin-right: 4px; }
  .result-excerpt {
    font-size: 14px;
    color: #4d5156;
    line-height: 1.55;
    margin-bottom: 8px;
  }
  .result-footer {
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: #70757a;
  }
</style>
</head>
<body>

<h1>IR Benchmark: Technical Documentation Search</h1>

<form class="search-form" id="searchForm">
  <input class="input-query" id="qInput" placeholder="Search technical documentation, flags, code, or architecture..." autofocus autocomplete="off">
  <select class="select-method" id="methodSelect">
    <option value="hybrid-rrf">Hybrid RRF (BM25F + Routed Neural)</option>
    <option value="bm25f">Hierarchical BM25F (Fields)</option>
    <option value="bm25">Flat BM25 (Lucene Default)</option>
    <option value="neural-routed">Cluster-Routed Neural (Soft Boundary)</option>
    <option value="neural-dense">Dense Cosine (Exhaustive)</option>
  </select>
  <button class="btn-search" type="submit">Search</button>
</form>

<div id="statusContainer"></div>
<main class="results-container" id="resultsContainer"></main>

<script>
const form = document.getElementById('searchForm');
const qInput = document.getElementById('qInput');
const methodSelect = document.getElementById('methodSelect');
const statusContainer = document.getElementById('statusContainer');
const resultsContainer = document.getElementById('resultsContainer');

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const query = qInput.value.trim();
  if (!query) return;

  statusContainer.innerHTML = '<span class="status-chip">Searching documentation...</span>';
  resultsContainer.innerHTML = '';

  const params = new URLSearchParams({ q: query, method: methodSelect.value });
  try {
    const res = await fetch('/api/search?' + params);
    const data = await res.json();
    if (!res.ok) {
      statusContainer.innerHTML = `<span class="status-chip error">Error: ${escapeHtml(data.error || 'Search request failed')}</span>`;
      return;
    }

    statusContainer.innerHTML = `<span class="status-chip">⚡ <b>${data.results.length} results</b> retrieved in <b>${data.elapsed_ms} ms</b> · Method: <b>${escapeHtml(data.method_name)}</b></span>`;

    if (data.results.length === 0) {
      resultsContainer.innerHTML = '<p style="color:#5f6368">No matching technical documents found for this query.</p>';
      return;
    }

    resultsContainer.innerHTML = data.results.map(x => {
      const headingHtml = x.heading_snippet ? `<div class="heading-snippet"><span class="crumb-icon">›</span> ${escapeHtml(x.heading_snippet)}</div>` : '';
      const techHtml = x.technology ? `<span class="tech-badge">${escapeHtml(x.technology)}</span>` : '';

      return `
        <article class="result-card">
          <div class="result-header">
            <span class="rank-badge">${x.rank}.</span>
            <a class="result-title" href="${x.url}" target="_blank" rel="noopener">${escapeHtml(x.title || 'Untitled Document')}</a>
          </div>
          <div class="result-url-bar">
            ${techHtml}
            <span class="url-text">${escapeHtml(x.url)}</span>
          </div>
          ${headingHtml}
          <div class="result-excerpt">${escapeHtml(x.excerpt)}</div>
          <div class="result-footer">
            <span>Relevance Score: <b>${x.score}</b></span>
            <span>Document ID: <code>${x.doc_id}</code></span>
          </div>
        </article>
      `;
    }).join('');

  } catch (err) {
    statusContainer.innerHTML = `<span class="status-chip error">Connection error: ${escapeHtml(err.message)}</span>`;
  }
};
</script>

</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--neural-output", type=Path, default=ROOT / "neural/outputs")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data/00_corpus")
    parser.add_argument(
        "--corpus-zip",
        type=Path,
        default=ROOT / "filter/outputs/stage_05_archive/technical_web_corpus.zip",
    )
    args = parser.parse_args()
    serve(args.host, args.port, args.neural_output, args.corpus_dir, args.corpus_zip)
