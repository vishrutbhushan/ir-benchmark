"""Load fixed result snapshots and persist shared query/document judgments."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from zipfile import ZipFile
from .common import ROOT, COLUMNS, read_csv, write_csv, write_json
from .metrics import calculate


class Store:
    def __init__(self, archive=None):
        self.lock = threading.RLock()
        self.queries = [
            json.loads(line)
            for line in (ROOT / "queries/final_queries.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len(self.queries) != 20 or len({q["query_id"] for q in self.queries}) != 20:
            raise ValueError("Need exactly 20 unique queries")
        texts = {q["query_id"]: q["query_text"] for q in self.queries}
        self.runs = {}
        self.documents = {}
        self.pool = set()
        for engine in ("classical", "neural"):
            for path in sorted((ROOT / "results" / engine).glob("*.csv")):
                if path.name == "timings.csv":
                    continue
                rows = read_csv(path)
                if not rows:
                    continue
                variant = rows[0]["variant"]
                if len(rows) != 200:
                    raise ValueError(f"{path.name}: need 200 results")
                grouped = defaultdict(list)
                for r in rows:
                    if (
                        r["engine"] != engine
                        or r["variant"] != variant
                        or r["query_text"] != texts.get(r["query_id"])
                    ):
                        raise ValueError("Result identity mismatch")
                    grouped[r["query_id"]].append(r)
                    old = self.documents.get(r["doc_id"])
                    if old and old["url"] != r["url"]:
                        raise ValueError(
                            "Document ID has conflicting URLs across snapshots"
                        )
                    self.documents[r["doc_id"]] = r
                    self.pool.add((r["query_id"], r["doc_id"]))
                if set(grouped) != set(texts):
                    raise ValueError("A run is missing queries")
                if any(
                    sorted(int(r["rank"]) for r in group) != list(range(1, 11))
                    or len({r["doc_id"] for r in group}) != 10
                    for group in grouped.values()
                ):
                    raise ValueError("Invalid top-ten ordering")
                self.runs[(engine, variant)] = rows
        if not all(
            any(e == engine for e, _ in self.runs) for engine in ("classical", "neural")
        ):
            raise ValueError("Both engine families need result snapshots")
        self.path = ROOT / "data/judgments.csv"
        self.labels = {key: None for key in self.pool}
        self.updated = {}
        if self.path.exists():
            for r in read_csv(self.path):
                key = (r["query_id"], r["doc_id"])
                if key not in self.pool:
                    raise ValueError("Judgments belong to a different result pool")
                if r["relevant"] not in {"", "0", "1"}:
                    raise ValueError("Invalid judgment")
                self.labels[key] = int(r["relevant"]) if r["relevant"] else None
                self.updated[key] = r.get("updated_at", "")
        self.archive = Path(archive) if archive else None
        self.paths = {}
        if self.archive and self.archive.is_file():
            with ZipFile(self.archive) as source:
                with source.open(
                    "technical_web_corpus/metadata/corpus_manifest.jsonl"
                ) as stream:
                    for line in stream:
                        d = json.loads(line)
                        if d["doc_id"] in self.documents:
                            if d["url"] != self.documents[d["doc_id"]]["url"]:
                                raise ValueError(
                                    "Archive and result document URLs differ"
                                )
                            self.paths[d["doc_id"]] = d["html_path"]
        if set(self.runs) != {("classical", "bm25"), ("neural", "dense")}:
            raise ValueError(
                "Keep only BM25 and dense snapshots in the active results folders"
            )
        self.save()

    def judgment_rows(self):
        return [
            {
                "query_id": q,
                "doc_id": d,
                "relevant": self.labels[(q, d)],
                "updated_at": self.updated.get((q, d), ""),
            }
            for q, d in sorted(self.pool)
        ]

    def save(self):
        write_csv(
            self.path,
            self.judgment_rows(),
            ["query_id", "doc_id", "relevant", "updated_at"],
        )

    def judge(self, qid, doc_id, value):
        key = qid, doc_id
        if key not in self.pool or not (
            value is None or (type(value) is int and value in (0, 1))
        ):
            raise ValueError("Unknown pair or invalid label")
        with self.lock:
            self.labels[key] = value
            self.updated[key] = datetime.now(timezone.utc).isoformat()
            self.save()

    def query(self, engine, variant, qid):
        key = engine, variant
        if key not in self.runs:
            raise ValueError("Unknown engine or variant")
        rows = []
        for r in self.runs[key]:
            if r["query_id"] == qid:
                rows.append(
                    {
                        **r,
                        "relevant": self.labels[(qid, r["doc_id"])],
                        "saved_url": f"/document/{r['doc_id']}"
                        if r["doc_id"] in self.paths
                        else None,
                    }
                )
        if not rows:
            raise ValueError("Unknown query")
        rows.sort(key=lambda r: int(r["rank"]))
        pool = [key for key in self.pool if key[0] == qid]
        return {
            "results": rows,
            "judged": sum(r["relevant"] is not None for r in rows),
            "pool_judged": sum(self.labels[k] is not None for k in pool),
            "pool_size": len(pool),
        }

    def exports(self, kind):
        if kind == "judgments":
            return self.judgment_rows(), [
                "query_id",
                "doc_id",
                "relevant",
                "updated_at",
            ]
        if kind == "results":
            rows = [
                {**r, "relevant": self.labels[(r["query_id"], r["doc_id"])]}
                for results in self.runs.values()
                for r in results
            ]
            return rows, COLUMNS + ["relevant"]
        if kind == "timings":
            rows = []
            for engine in ("classical", "neural"):
                path = ROOT / f"results/{engine}/timings.csv"
                if path.exists():
                    rows.extend(read_csv(path))
            return rows, list(rows[0]) if rows else []
        details, summary = calculate(self.runs, self.labels)
        rows = summary if kind == "metrics" else details
        return rows, list(rows[0])

    def html(self, doc_id):
        if doc_id not in self.paths:
            raise KeyError(doc_id)
        with ZipFile(self.archive) as source:
            return source.read("technical_web_corpus/" + self.paths[doc_id]).decode(
                "utf-8", errors="replace"
            )

    def comparison_report(self):
        classical = json.loads((ROOT / "results/classical/manifest.json").read_text())
        neural = json.loads((ROOT / "results/neural/manifest.json").read_text())
        missing = {
            f"{engine}/{variant}": len(
                {r["doc_id"] for r in rows if r["doc_id"] not in self.paths}
            )
            for (engine, variant), rows in self.runs.items()
        }
        report = {
            "classical_corpus_documents": classical[0]["corpus_documents"],
            "neural_corpus_documents": neural[0]["corpus_documents"],
            "controlled_comparison": False,
            "reason": "Classical export uses a different corpus and does not provide its manifest hash",
            "unique_query_document_pairs": len(self.pool),
            "result_rows": sum(len(r) for r in self.runs.values()),
            "saved_html_unavailable_unique_documents_by_variant": missing
            if self.archive
            else None,
            "metrics_scope": "Recall, F1, AP and NDCG use the union of all exported top-ten results as the judged pool; full-corpus recall and full MAP are unavailable",
            "judgment_policy": "Unjudged stays blank. One query/document judgment applies across all variants.",
        }
        write_json(ROOT / "results/comparison.json", report)
        return report
