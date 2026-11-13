"""Read the six saved result sets and strict pooled judgments."""

import json
from collections import defaultdict

from config import load_pipeline_config, resolve_path

from .common import ROOT, read_csv, write_json
from .metrics import calculate, calculate_graded


class Store:
    def __init__(self):
        self.config = load_pipeline_config()
        self.results_dir = resolve_path(self.config["paths"]["evaluation_results_dir"])
        self.queries = [
            json.loads(line)
            for line in resolve_path(self.config["paths"]["queries"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if not self.queries or len({q["query_id"] for q in self.queries}) != len(self.queries):
            raise ValueError("The configured query set must have unique query IDs")
        self.query_text = {q["query_id"]: q["query_text"] for q in self.queries}
        known=json.loads((resolve_path(self.config["paths"]["judgments"]).parent/"judgment_queries.json").read_text())
        if any(qid in known and known[qid]!=text for qid,text in self.query_text.items()):
            raise ValueError("Query text changed under a judged query ID")
        self.query_types = {q["query_id"]: q.get("query_type", "unspecified") for q in self.queries}
        self.runs = {}
        self.documents = {}
        self.pool = set()
        limit = int(self.config["evaluation"]["results_per_query"])

        for item in self.config["evaluation"]["result_snapshots"]:
            path = self.results_dir / item["path"]
            rows = read_csv(path)
            expected = len(self.queries) * limit
            if len(rows) != expected:
                raise ValueError(f"{path}: expected {expected} result rows, found {len(rows)}")
            self._validate_run(rows, item["engine"], item["variant"], limit)
            for row in rows:
                self.documents[row["doc_id"]] = row
                self.pool.add((row["query_id"], row["doc_id"]))
            self.runs[(item["engine"], item["variant"])] = rows

        self.labels = {}
        self.grades = {}
        judgment_path = resolve_path(self.config["paths"]["judgments"])
        seen = set()
        for row in read_csv(judgment_path):
            key = (row["query_id"], row["doc_id"])
            if key in seen:
                raise ValueError("Duplicate master judgment pair")
            seen.add(key)
            if key not in self.pool:
                continue
            if row.get("relevant") not in (None, ""):
                self.labels[key] = int(row["relevant"])
            if row.get("consensus_grade") not in (None, ""):
                self.grades[key] = int(row["consensus_grade"])

        missing = self.pool - self.labels.keys()
        missing_grades = self.pool - self.grades.keys()
        if missing or missing_grades:
            raise ValueError(
                f"Judgments are incomplete: {len(missing)} binary labels and "
                f"{len(missing_grades)} grades missing from the six-engine pool"
            )

    def _validate_run(self, rows, engine, variant, limit):
        grouped = defaultdict(list)
        for row in rows:
            if row["engine"] != engine or row["variant"] != variant:
                raise ValueError(f"Unexpected engine identity in {engine}/{variant}")
            if row["query_text"] != self.query_text.get(row["query_id"]):
                raise ValueError(f"Query text mismatch in {engine}/{variant}")
            grouped[row["query_id"]].append(row)
        if set(grouped) != set(self.query_text):
            raise ValueError(f"{engine}/{variant} does not include all configured queries")
        for query_id, results in grouped.items():
            ranks = sorted(int(row["rank"]) for row in results)
            if ranks != list(range(1, limit + 1)) or len({r["doc_id"] for r in results}) != limit:
                raise ValueError(f"Invalid top-{limit} results for {engine}/{variant}/{query_id}")

    def exports(self, kind):
        if kind == "metrics_graded":
            details, summary = calculate_graded(self.runs, self.labels, self.grades)
            rows = summary
        elif kind == "per_query_graded":
            rows, _ = calculate_graded(self.runs, self.labels, self.grades)
        elif kind == "metrics":
            details, summary = calculate(self.runs, self.labels, self.query_types)
            rows = summary
        elif kind == "per_query":
            rows, _ = calculate(self.runs, self.labels, self.query_types)
        else:
            raise ValueError(f"Unsupported evaluation export: {kind}")
        return rows, list(rows[0]) if rows else []

    def comparison_report(self):
        manifests = {}
        for engine in ("classical", "neural"):
            path = self.results_dir / engine / "manifest.json"
            if path.exists():
                manifests[engine] = json.loads(path.read_text(encoding="utf-8"))
        classical = manifests.get("classical", [{}])[0]
        neural = manifests.get("neural", [{}])[0]
        same_corpus = (
            classical.get("corpus_documents") == neural.get("corpus_documents")
            and classical.get("corpus_manifest_sha256") is not None
            and classical.get("corpus_manifest_sha256") == neural.get("corpus_manifest_sha256")
        )
        report = {
            "classical_corpus_documents": classical.get("corpus_documents"),
            "neural_corpus_documents": neural.get("corpus_documents"),
            "controlled_comparison": same_corpus,
            "reason": None if same_corpus else "Corpus counts or manifest hashes do not match",
            "active_engines": [f"{engine}/{variant}" for engine, variant in self.runs],
            "unique_query_document_pairs": len(self.pool),
            "result_rows": sum(len(rows) for rows in self.runs.values()),
            "metrics_scope": "Pooled metrics use strict labels for the six active engines; Binary NDCG is primary and graded NDCG is secondary.",
        }
        write_json(self.results_dir / "comparison.json", report)
        return report
