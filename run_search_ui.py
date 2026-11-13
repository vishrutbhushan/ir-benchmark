#!/usr/bin/env python3
"""Serve search results and saved HTML from the six retrieval engines."""

import html
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs,urlsplit
from config import load_pipeline_config
from retrieval import RetrievalRuntime
from neural.src.common.gpu_lock import gpu_job_lock

PIPELINE_CONFIG=load_pipeline_config()
SEARCH_ENGINES=PIPELINE_CONFIG["evaluation"]["search_engines"]
METHOD_NAMES={x["id"]:x["name"] for x in SEARCH_ENGINES}

def get_configured_limit():
    return PIPELINE_CONFIG["neural"]["result_limit"]

def serve(host, port):
    runtime = RetrievalRuntime()
    with gpu_job_lock():
        runtime.load()
        runtime.neural.warmup()
    print(f"Search engines ready at http://{host}:{port}",flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[search-ui] {fmt % args}", flush=True)

        def respond(self, payload, status=200, content_type="application/json", headers=None):
            try:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self.respond(UI.encode(), content_type="text/html")
                return
            if parsed.path.startswith("/document/"):
                doc_id = parsed.path.removeprefix("/document/")
                if not re.fullmatch(r"document_[0-9]+", doc_id):
                    self.send_error(404)
                    return
                if not runtime.neural:
                    self.respond({"error": "Saved corpus is unavailable"}, 503)
                    return
                try:
                    source = runtime.neural.corpus.read_html(doc_id)
                except (KeyError, FileNotFoundError):
                    self.send_error(404)
                    return
                body = (
                    "<!doctype html><meta charset=utf-8><title>Saved document</title>"
                    '<iframe sandbox="" referrerpolicy="no-referrer" style="border:0;width:100%;height:97vh" srcdoc="'
                    + html.escape(source, quote=True)
                    + '"></iframe>'
                ).encode()
                self.respond(
                    body,
                    content_type="text/html",
                    headers={
                        "Content-Security-Policy": "default-src 'none'; frame-src 'self' about:; style-src 'unsafe-inline'; img-src data:; script-src 'none'; form-action 'none'; base-uri 'none'"
                    },
                )
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
                limit = min(int(limit_arg), get_configured_limit()) if limit_arg and limit_arg.isdigit() else get_configured_limit()
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
        runtime.close()
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
  .pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin: 18px 0 28px;
  }
  .pagination button {
    border: 1px solid var(--border);
    border-radius: 6px;
    background: white;
    color: var(--text);
    padding: 7px 12px;
    cursor: pointer;
  }
  .pagination button:disabled { opacity: 0.45; cursor: default; }
</style>
</head>
<body>

<h1>IR Benchmark: Technical Documentation Search</h1>

<form class="search-form" id="searchForm">
  <input class="input-query" id="qInput" placeholder="Search technical documentation, flags, code, or architecture..." autofocus autocomplete="off">
  <select class="select-method" id="methodSelect">{method_options}</select>
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
const PAGE_SIZE = {page_size};
const RESULT_LIMIT = {result_limit};
let rankedResults = [];
let currentPage = 1;

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderResults(page) {
  const pageCount = Math.max(1, Math.ceil(rankedResults.length / PAGE_SIZE));
  currentPage = Math.max(1, Math.min(page, pageCount));
  const first = (currentPage - 1) * PAGE_SIZE;
  const shown = rankedResults.slice(first, first + PAGE_SIZE);
  const cards = shown.map(x => {
    const techHtml = x.technology ? `<span class="tech-badge">${escapeHtml(x.technology)}</span>` : '';
    const categoryHtml = x.category ? `<span class="tech-badge">${escapeHtml(x.category)}</span>` : '';
    return `
      <article class="result-card">
        <div class="result-header">
          <span class="rank-badge">${x.rank}.</span>
          <a class="result-title" href="${escapeHtml(x.url)}" target="_blank" rel="noopener">${escapeHtml(x.title || 'Untitled Document')}</a>
        </div>
        <div class="result-url-bar">${techHtml}${categoryHtml}<span class="url-text">${escapeHtml(x.url)}</span></div>
        <div class="result-footer">
          <span>Relevance Score: <b>${x.score}</b></span>
          <span>Document ID: <code>${x.doc_id}</code></span>
          <a href="${escapeHtml(x.saved_html)}" target="_blank" rel="noopener">Open saved HTML</a>
        </div>
      </article>`;
  }).join('');
  const controls = pageCount > 1 ? `
    <nav class="pagination" aria-label="Search result pages">
      <button type="button" id="previousPage" ${currentPage === 1 ? 'disabled' : ''}>Previous</button>
      <span>Page ${currentPage} of ${pageCount}</span>
      <button type="button" id="nextPage" ${currentPage === pageCount ? 'disabled' : ''}>Next</button>
    </nav>` : '';
  resultsContainer.innerHTML = cards + controls;
  document.getElementById('previousPage')?.addEventListener('click', () => renderResults(currentPage - 1));
  document.getElementById('nextPage')?.addEventListener('click', () => renderResults(currentPage + 1));
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

    statusContainer.innerHTML = `<span class="status-chip"> <b>${data.results.length} results</b> retrieved in <b>${data.elapsed_ms} ms</b> · Method: <b>${escapeHtml(data.method_name)}</b></span>`;

    if (data.results.length === 0) {
      rankedResults = [];
      resultsContainer.innerHTML = '<p style="color:#5f6368">No matching technical documents found for this query.</p>';
      return;
    }
    rankedResults = data.results.slice(0, RESULT_LIMIT);
    renderResults(1);

  } catch (err) {
    statusContainer.innerHTML = `<span class="status-chip error">Connection error: ${escapeHtml(err.message)}</span>`;
  }
};
</script>

</body>
</html>
"""

UI = UI.replace(
    "{method_options}",
    "".join(
        f'<option value="{html.escape(item["id"], quote=True)}">{html.escape(item["name"])}</option>'
        for item in SEARCH_ENGINES
    ),
)
UI = UI.replace("{page_size}", str(PIPELINE_CONFIG["neural"]["page_size"]))
UI = UI.replace("{result_limit}", str(PIPELINE_CONFIG["neural"]["result_limit"]))

if __name__ == "__main__":
    defaults=PIPELINE_CONFIG["evaluation"]
    serve(defaults["search_host"],defaults["search_port"])
