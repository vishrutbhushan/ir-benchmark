"""Local-only search UI/API; pagination reuses cached ranked results."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from src.step_08_search.step_08_search import MODES


def serve(engine, host="127.0.0.1", port=8000):
    print("Preparing the search model and index...", flush=True)
    warmup_ms = engine.warmup()
    print(f"Search warmup complete: {warmup_ms:.0f} ms", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/search.js": (
                        "search.js",
                        "application/javascript; charset=utf-8",
                    ),
                    "/style.css": ("style.css", "text/css; charset=utf-8"),
                }
                if parsed.path in assets:
                    filename, content_type = assets[parsed.path]
                    data = (
                        Path(__file__).with_name("web").joinpath(filename).read_bytes()
                    )
                elif parsed.path.startswith("/document/"):
                    doc_id = parsed.path.removeprefix("/document/")
                    if not re.fullmatch(r"document_[0-9]+", doc_id):
                        self.send_error(404)
                        return
                    try:
                        source = engine.corpus.read_html(doc_id)
                    except (KeyError, FileNotFoundError):
                        self.send_error(404)
                        return
                    data = (
                        "<!doctype html><meta charset=utf-8><title>Saved document</title>"
                        '<iframe sandbox="" referrerpolicy="no-referrer" style="border:0;width:100%;height:97vh" srcdoc="'
                        + html.escape(source, quote=True)
                        + '"></iframe>'
                    ).encode()
                    content_type = "text/html; charset=utf-8"
                elif parsed.path == "/api/options":
                    data = json.dumps(
                        {
                            "documents": len(engine.documents),
                            "passages": sum(len(m) for m in engine.matrices.values()),
                            "modes": list(MODES),
                            "default_mode": engine.default_mode,
                            "weights": dict(
                                zip(
                                    ("title", "headings", "body", "code"),
                                    map(float, engine.weights),
                                )
                            ),
                            "result_limit": engine.config["result_limit"],
                            "page_size": engine.config["page_size"],
                            "index_version": engine.summary["fingerprint"],
                            "startup_warmup_ms": engine.startup_warmup_ms,
                        }
                    ).encode()
                    content_type = "application/json"
                elif parsed.path == "/api/search":
                    values = parse_qs(parsed.query)

                    def get(name, default=""):
                        return values.get(name, [default])[0]

                    page = int(get("page", "1"))
                    size = engine.config["page_size"]
                    maximum = (engine.config["result_limit"] + size - 1) // size
                    if not 1 <= page <= maximum:
                        raise ValueError(f"Page must be between 1 and {maximum}")
                    if set(values) - {"q", "mode", "page"}:
                        raise ValueError(
                            "Only q, mode and page parameters are supported"
                        )
                    result = engine.search(
                        get("q"),
                        mode=get("mode", engine.default_mode),
                        offset=(page - 1) * size,
                    )
                    result = {**result, "page": page}
                    data = json.dumps(result, ensure_ascii=False).encode()
                    content_type = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                if parsed.path.startswith("/document/"):
                    self.send_header(
                        "Content-Security-Policy",
                        "default-src 'none'; frame-src 'self' about:; style-src 'unsafe-inline'; img-src data:; script-src 'none'; form-action 'none'; base-uri 'none'",
                    )
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError) as error:
                data = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    server = HTTPServer((host, port), Handler)
    print(
        f"Technical search: http://{host}:{port} ({len(engine.documents):,} documents)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
