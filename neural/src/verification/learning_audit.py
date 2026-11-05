"""Audit reviewed binary zone-weight inputs."""

from collections import Counter
from contextlib import closing
import csv
import json
import math
from pathlib import Path

from src.common.artifacts import connect, digest_file, report
from src.common.config import load_config
from src.learning.features import load_examples
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
        settings.get("weight_queries", "data_inputs/neural_weights/zone_weight_queries.csv")
    )
    judgments_path = _configured_input(
        settings.get(
            "weight_judgments",
            "neural/data/zone_weight_judgments.csv",
        )
    )
    relevance, queries = load_examples(queries_path, judgments_path)
    features = read_csv(
        REPOSITORY_ROOT
        / "neural/results/step_04_weights/features/training_features.csv"
    )
    feature_map = {(r["query_id"], r["doc_id"]): r for r in features}
    expected_pairs = {(r["query_id"], r["doc_id"]) for r in relevance}
    if expected_pairs != set(feature_map):
        raise ValueError("Generated zone features do not match judgments")
    expected_queries = {q["query_id"]: q for q in queries}
    with closing(connect(root / "step_01_extraction/corpus.sqlite", True)) as db:
        for row in relevance:
            if row["query_id"] not in expected_queries:
                raise ValueError("Judgment references an unknown query")
            source = db.execute(
                "SELECT record_json FROM documents WHERE doc_id=?", (row["doc_id"],)
            ).fetchone()
            if source is None:
                raise ValueError(f"Unknown zone-weight document: {row['doc_id']}")
            doc = json.loads(source[0])
            evidence = (
                "\n".join(doc["headings"])
                if row["evidence_zone"] == "headings"
                else doc[
                    {"title": "title", "body": "body_text", "code": "code_text"}[
                        row["evidence_zone"]
                    ]
                ]
            )
            if not row["evidence_text"].strip() or not evidence.strip():
                raise ValueError("Zone-weight evidence must be nonempty")
            grade = float(row["relevance_grade"])
            if grade not in {0.0, 1.0}:
                raise ValueError("Zone-weight grades must be binary 0 or 1")
            feature = feature_map[(row["query_id"], row["doc_id"])]
            for zone in ("title", "headings", "body", "code"):
                present, score = feature[f"has_{zone}"], feature[f"{zone}_cosine"]
                if present not in {"0", "1"} or (
                    present == "0" and score != ""
                ):
                    raise ValueError("Invalid missing-zone representation")
                if present == "1" and not -1.00001 <= float(score) <= 1.00001:
                    raise ValueError("Invalid cosine feature")
    grades = [float(row["relevance_grade"]) for row in relevance]
    stats = {
        "checks_passed": True,
        "zone_pairs": len(relevance),
        "zone_queries": len(queries),
        "grade_distribution": dict(Counter(f"{grade:.2f}" for grade in grades)),
        "zone_queries_sha256": digest_file(queries_path),
        "zone_judgments_sha256": digest_file(judgments_path),
    }
    report(root / "step_07_verification/learning_audit", "Neural learning-data audit", stats, {})
    print(json.dumps(stats, indent=2), flush=True)
    return stats
