"""Check learning tables, query separation and source-document consistency."""

from collections import Counter
from contextlib import closing
import csv
import json
import math
from pathlib import Path

from src.common.artifacts import connect, digest_file, report
from src.step_05_classification.labels import canonical_family, load_training_examples
from src.step_04_weights.features import load_examples
from src.common.zones import ZONES


def read_csv(path):
    csv.field_size_limit(16 * 1024 * 1024)
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def audit_learning_data(root):
    root = Path(root)
    base = Path("data")
    relevance = load_examples(base / "step_04_weights")
    features = read_csv("results/step_04_weights/features/training_features.csv")
    feature_map = {(r["query_id"], r["doc_id"]): r for r in features}
    classes = read_csv(base / "step_05_classification/classes.csv")
    classified = load_training_examples(
        root, base / "step_05_classification/training.csv"
    )
    if not classes or not classified:
        raise ValueError("Classification tables must not be empty")
    class_names = {r["class_id"]: r["class_name"] for r in classes}
    if len(class_names) != len(classes):
        raise ValueError("Duplicate class/technology definitions")
    if len(relevance) != len({(r["query_id"], r["doc_id"]) for r in relevance}):
        raise ValueError("Duplicate relevance pairs")
    query_table = read_csv(base / "step_04_weights/queries.csv")
    intent_splits = {}
    for q in query_table:
        if q["split"] != intent_splits.setdefault(q["intent_id"], q["split"]):
            raise ValueError("Related query formulations cross learning splits")
    expected_queries = {
        q["query_id"]: (q["query_text"], q["split"]) for q in query_table
    }
    query_splits = {}
    query_labels = {}
    with closing(connect(root / "step_01_extraction/corpus.sqlite", True)) as db:

        def document(doc_id):
            row = db.execute(
                "SELECT record_json FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown labeled document: {doc_id}")
            return json.loads(row[0])

        for row in relevance:
            qid = row["query_id"]
            signature = (row["query_text"], row["split"])
            if signature != query_splits.setdefault(qid, signature):
                raise ValueError("Query text/split inconsistency")
            doc = document(row["doc_id"])
            if (row["doc_title"], row["doc_url"], row["doc_technology"]) != (
                doc["title"],
                doc["url"],
                doc["metadata"]["technology"],
            ):
                raise ValueError("Judgment source metadata changed")
            if row["use_for_weight_learning"] == "1":
                query_labels.setdefault(qid, set()).add(row["relevant"])
            evidence = doc[
                {"title": "title", "body": "body_text", "code": "code_text"}.get(
                    row["evidence_zone"], "body_text"
                )
            ]
            if (
                not 0
                <= int(row["evidence_start"])
                < int(row["evidence_end"])
                <= len(evidence)
                or not row["rationale"].strip()
            ):
                raise ValueError("Judgment evidence or rationale is missing")
            feature = feature_map[(row["query_id"], row["doc_id"])]
            for zone in ZONES:
                present, score = feature[f"has_{zone}"], feature[f"{zone}_cosine"]
                if present not in {"0", "1"} or (present == "0" and score != ""):
                    raise ValueError("Invalid missing-zone representation")
                if present == "1" and (
                    not math.isfinite(float(score))
                    or not -1.00001 <= float(score) <= 1.00001
                ):
                    raise ValueError("Invalid cosine feature")
        if any(query_labels.get(qid) != {"0", "1"} for qid in query_splits):
            raise ValueError("Every query needs usable positive and negative examples")
        if len(classified) != len({r["doc_id"] for r in classified}):
            raise ValueError("Duplicate classification documents")
        families, content = {}, {}
        for row in classified:
            doc = document(row["doc_id"])
            if (
                row["class_id"] not in class_names
                or row["class_name"] != class_names[row["class_id"]]
            ):
                raise ValueError("Invalid reviewed content class")
            for mapping, key in (
                (families, canonical_family(doc["url"])),
                (content, doc["metadata"]["content_sha256"]),
            ):
                if row["split"] != mapping.setdefault(key, row["split"]):
                    raise ValueError(
                        "Duplicate family/content crosses classification splits"
                    )
    if query_splits != expected_queries:
        raise ValueError("Query table and relevance judgments disagree")
    split_counts = Counter((r["class_id"], r["split"]) for r in classified)
    if len(split_counts) != len(classes) * 2 or any(
        n != {"train": 10, "validation": 2}[split]
        for (_, split), n in split_counts.items()
    ):
        raise ValueError("Classification learning subset is not balanced as specified")
    stats = {
        "checks_passed": True,
        "zone_pairs": len(relevance),
        "usable_zone_pairs": sum(int(r["use_for_weight_learning"]) for r in relevance),
        "development_queries": len(query_splits),
        "query_splits": dict(Counter(split for _, split in query_splits.values())),
        "classification_documents": len(classified),
        "classes": len(classes),
        "classification_splits": dict(Counter(r["split"] for r in classified)),
        "query_scope": "Learning queries only; final benchmark queries live on the evaluation branch",
        "source_text_and_content_labels_checked": True,
        "classification_duplicate_group_leakage": False,
        "semantic_label_scope": "Binary query-specific development relevance; classification labels describe reviewed primary content topics. Structural checks do not prove judgment correctness.",
        "zone_training_sha256": digest_file(base / "step_04_weights/training.csv"),
        "classification_training_sha256": digest_file(
            base / "step_05_classification/training.csv"
        ),
    }
    report(
        root / "step_07_verification/learning_audit",
        "Neural learning-data audit",
        stats,
        {},
    )
    print(json.dumps(stats, indent=2), flush=True)
    return stats
