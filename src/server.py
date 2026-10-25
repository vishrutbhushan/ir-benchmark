"""Serve fixed retrieval results and save binary judgments."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import html
import json
import logging
from urllib.parse import parse_qs, urlsplit
from .common import ROOT, csv_text


def serve(store, host, port):
    comparison = store.comparison_report()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logging.info(format, *args)

        def respond(
            self,
            data,
            content_type="application/json; charset=utf-8",
            status=200,
            filename=None,
            saved=False,
        ):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
            if saved:
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; frame-src 'self' about:; style-src 'unsafe-inline'; img-src data:; script-src 'none'; form-action 'none'; base-uri 'none'",
                )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlsplit(self.path)
            values = parse_qs(parsed.query)
            get = lambda key, default="": values.get(key, [default])[0]
            try:
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "application/javascript"),
                    "/style.css": ("style.css", "text/css"),
                }
                if parsed.path in assets:
                    name, mime = assets[parsed.path]
                    self.respond((ROOT / "web" / name).read_bytes(), mime)
                elif parsed.path == "/api/options":
                    self.respond(
                        {
                            "queries": store.queries,
                            "variants": {
                                e: [v for family, v in store.runs if family == e]
                                for e in ("classical", "neural")
                            },
                            "comparison": comparison,
                            "judged": sum(v is not None for v in store.labels.values()),
                            "pool_size": len(store.pool),
                        }
                    )
                elif parsed.path == "/api/query":
                    self.respond(
                        store.query(get("engine"), get("variant"), get("query_id"))
                    )
                elif parsed.path == "/api/export":
                    kind = get("kind", "judgments")
                    if kind == "qrels":
                        data = "".join(
                            f"{q}\t0\t{d}\t{value}\n"
                            for (q, d), value in sorted(store.labels.items())
                            if value is not None
                        )
                        self.respond(
                            data,
                            "text/tab-separated-values; charset=utf-8",
                            filename="qrels.tsv",
                        )
                    elif kind in {
                        "judgments",
                        "results",
                        "metrics",
                        "per_query",
                        "timings",
                    }:
                        rows, columns = store.exports(kind)
                        self.respond(
                            csv_text(rows, columns),
                            "text/csv; charset=utf-8",
                            filename=kind + ".csv",
                        )
                    else:
                        raise ValueError("Unknown export")
                elif parsed.path.startswith("/document/"):
                    try:
                        source = store.html(parsed.path.removeprefix("/document/"))
                    except KeyError:
                        self.send_error(404)
                        return
                    page = (
                        '<!doctype html><meta charset=utf-8><title>Saved document</title><iframe sandbox="" referrerpolicy="no-referrer" style="border:0;width:100%;height:97vh" srcdoc="'
                        + html.escape(source, quote=True)
                        + '"></iframe>'
                    )
                    self.respond(page, "text/html; charset=utf-8", saved=True)
                else:
                    self.send_error(404)
            except (ValueError, KeyError) as error:
                self.respond({"error": str(error)}, status=400)

        def do_POST(self):
            if self.path != "/api/judgment":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                store.judge(body["query_id"], body["doc_id"], body["relevant"])
                self.respond({"saved": True})
            except (ValueError, KeyError) as error:
                self.respond({"error": str(error)}, status=400)

    server = HTTPServer((host, port), Handler)
    print(f"Evaluation UI: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
