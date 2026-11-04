#!/usr/bin/env python3
"""Serve a simple, read-only comparison search UI."""

import argparse
import html
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent


class SearchRuntime:
    def __init__(self, neural_output, corpus_dir, corpus_zip):
        self.classical = {}
        self.neural = None
        self.neural_output = neural_output
        self.corpus_dir = corpus_dir
        self.corpus_zip = corpus_zip

    def _load_classical(self):
        if self.classical:
            return
        classical = ROOT / "classical"
        # Temporarily clear any conflicting src modules
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
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from src.retrieval.step_08_search import SearchEngine

            self.neural = SearchEngine(
                self.neural_output,
                device,
                self.corpus_dir,
                corpus_zip=self.corpus_zip,
            )
        finally:
            # Keep neural in sys.path/sys.modules for SearchEngine operations
            pass

    def search(self, method, query):
        started = time.perf_counter()
        if method in {"bm25", "bm25f"}:
            self._load_classical()
            payload = self.classical[method].search(query, top_k=10)
            rows = payload["results"]
        else:
            self._load_neural()
            mode = "routed-cluster" if method == "neural-routed" else "dense"
            rows = self.neural.search(query, mode=mode, page_size=10)["results"]
        return {
            "method": method,
            "query": query,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": [self._row(row, rank) for rank, row in enumerate(rows, 1)],
        }

    @staticmethod
    def _row(row, rank):
        return {
            "rank": rank,
            "doc_id": row.get("doc_id", ""),
            "title": row.get("title", ""),
            "technology": row.get("technology", ""),
            "category": row.get("category", row.get("categories", "")),
            "score": round(float(row.get("score", 0)), 6),
            "excerpt": row.get("snippet", row.get("body_text", ""))[:400],
            "url": row.get("url", ""),
        }


def serve(host, port, neural_output, corpus_dir, corpus_zip):
    print("Preloading search engines into memory...", flush=True)
    runtime = SearchRuntime(neural_output, corpus_dir, corpus_zip)
    print("  [1/2] Loading Classical inverted index into memory...", flush=True)
    runtime._load_classical()
    print("  [2/2] Loading Neural search engine & model weights into memory...", flush=True)
    runtime._load_neural()
    print("  Warming up search caches...", flush=True)
    if runtime.neural is not None:
        runtime.neural.warmup()
    print("Ready! All search engines are resident in memory.\n", flush=True)

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
                method = values.get("method", ["bm25"])[0]
                if not query or method not in {"bm25", "bm25f", "neural-dense", "neural-routed"}:
                    self.respond({"error": "Provide q and a supported method"}, 400)
                    return
                try:
                    self.respond(runtime.search(method, query))
                except (RuntimeError, ValueError, FileNotFoundError) as error:
                    logging.exception("Search request failed")
                    self.respond({"error": str(error)}, 503)
                return
            self.send_error(404)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Search UI: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if runtime.neural is not None:
            runtime.neural.close()
        server.server_close()


UI = """<!doctype html>
<meta charset="utf-8">
<title>IR Benchmark Search</title>
<style>
body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#17202a}
form{display:flex;gap:8px}input{flex:1;padding:12px;font-size:16px}select,button{padding:12px}
article{border-top:1px solid #ddd;padding:16px 0}small{color:#5d6d7e}a{color:#1f618d}
</style>
<h1>IR Benchmark Search</h1>
<form id="form"><input id="q" placeholder="Search technical documentation" autofocus>
<select id="method"><option value="bm25">Classical: Flat BM25</option>
<option value="bm25f">Classical: Hierarchical BM25F</option>
<option value="neural-dense">Neural: Dense Cosine</option>
<option value="neural-routed">Neural: Routed Cluster</option></select>
<button>Search</button></form><p id="status"></p><main id="results"></main>
<script>
form.onsubmit=async e=>{e.preventDefault();status.textContent="Searching...";results.innerHTML="";
let p=new URLSearchParams({q:q.value,method:method.value});let r=await fetch("/api/search?"+p);
let d=await r.json();if(!r.ok){status.textContent=d.error;return}
status.textContent=`${d.results.length} results in ${d.elapsed_ms} ms`;
results.innerHTML=d.results.map(x=>`<article><b>${x.rank}. ${escapeHtml(x.title)}</b>
<br><small>${escapeHtml(x.technology)} · ${escapeHtml(x.category)} · score ${x.score}</small>
<p>${escapeHtml(x.excerpt)}</p><a href="${x.url}" target="_blank" rel="noopener">Original Source</a></article>`).join("")};
function escapeHtml(s){return String(s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
</script>"""


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
