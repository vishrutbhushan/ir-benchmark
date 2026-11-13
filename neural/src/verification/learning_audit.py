"""Audit reviewed binary zone-weight inputs."""

from collections import Counter
from contextlib import closing
import csv
import json
import math
from pathlib import Path

from src.common.artifacts import connect, digest_file, report
from src.common.config import load_config
from src.common.paths import REPOSITORY_ROOT


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _configured_input(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def audit_learning_data(root):
    root = Path(root)
    settings = load_config()
    queries_path = _configured_input(
        settings.get("weight_queries", "neural/data/zone_weight_queries.csv")
    )
    judgments_path = _configured_input(
        settings.get(
            "weight_judgments",
            "neural/data/zone_weight_judgments.csv",
        )
    )
    queries = read_csv(queries_path)
    relevance = read_csv(judgments_path)
    query_map = {row["query_id"]: row["query_text"] for row in queries}
    if len(query_map) != len(queries) or not query_map:
        raise ValueError("Zone-weight queries must have unique IDs")
    required = {"query_id", "query_text", "doc_id", "doc_title", "s_T", "s_H", "s_B", "s_C", "relevant_r"}
    if not relevance or not required.issubset(relevance[0]):
        raise ValueError("Zone-weight judgments have an invalid schema")
    seen = set()
    grades = []
    with closing(connect(root / "step_01_extraction/corpus.sqlite", True)) as db:
        for row in relevance:
            pair = (row["query_id"], row["doc_id"])
            if pair in seen:
                raise ValueError("Zone-weight judgments contain duplicate pairs")
            seen.add(pair)
            if row["query_id"] not in query_map or row["query_text"] != query_map[row["query_id"]]:
                raise ValueError("Judgment references an unknown query")
            source = db.execute(
                "SELECT title FROM documents WHERE doc_id=?", (row["doc_id"],)
            ).fetchone()
            if source is None:
                raise ValueError(f"Unknown zone-weight document: {row['doc_id']}")
            if row["doc_title"].strip() != (source[0] or "").strip():
                raise ValueError(f"Zone-weight title mismatch for {row['doc_id']}")
            grade = float(row["relevant_r"])
            if grade not in {0.0, 1.0}:
                raise ValueError("Zone-weight grades must be binary 0 or 1")
            grades.append(grade)
            for zone in ("s_T", "s_H", "s_B", "s_C"):
                value = float(row[zone])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("Zone-weight features must be finite values in [0, 1]")
    stats = {
        "checks_passed": True,
        "zone_pairs": len(relevance),
        "zone_queries": len(query_map),
        "grade_distribution": dict(Counter(f"{grade:.2f}" for grade in grades)),
        "zone_queries_sha256": digest_file(queries_path),
        "zone_judgments_sha256": digest_file(judgments_path),
    }
    report(root / "step_07_verification/learning_audit", "Neural learning-data audit", stats, {})
    print(json.dumps(stats, indent=2), flush=True)
    return stats
