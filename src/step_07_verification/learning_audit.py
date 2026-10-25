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
    classes = read_csv(base / "step_05_classification/classes.csv")
    mappings = read_csv(base / "step_05_classification/technology_classes.csv")
    classified = load_training_examples(
        root, base / "step_05_classification/training.csv"
    )
    if not classes or not mappings or not classified:
        raise ValueError("Classification tables must not be empty")
    class_names = {r["class_id"]: r["class_name"] for r in classes}
    technology_class = {r["technology"]: r["class_id"] for r in mappings}
    if len(class_names) != len(classes) or len(technology_class) != len(mappings):
        raise ValueError("Duplicate class/technology definitions")
    if len(relevance) != len({(r["query_id"], r["doc_id"]) for r in relevance}):
        raise ValueError("Duplicate relevance pairs")
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
            for zone in ZONES:
                present, score = row[f"has_{zone}"], row[f"{zone}_cosine"]
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
            cid = technology_class[doc["metadata"]["technology"]]
            if (row["class_id"], row["class_name"]) != (cid, class_names[cid]):
                raise ValueError("Class label differs from source-domain mapping")
            for mapping, key in (
                (families, canonical_family(doc["url"])),
                (content, doc["metadata"]["content_sha256"]),
            ):
                if row["split"] != mapping.setdefault(key, row["split"]):
                    raise ValueError(
                        "Duplicate family/content crosses classification splits"
                    )
        corpus_technologies = {
            r[0] for r in db.execute("SELECT DISTINCT technology FROM documents")
        }
        if corpus_technologies != set(technology_class):
            raise ValueError("Class mapping does not cover the corpus technologies")
    split_counts = Counter((r["class_id"], r["split"]) for r in classified)
    if len(split_counts) != len(classes) * 3 or any(
        n != {"train": 20, "validation": 5, "test": 5}[split]
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
        "source_text_and_label_mapping_checked": True,
        "classification_duplicate_group_leakage": False,
        "semantic_label_scope": "Binary query-specific development relevance; classification labels describe source domains. Structural checks do not prove judgment correctness.",
        "zone_training_sha256": digest_file(base / "step_04_weights/training.csv"),
        "classification_training_sha256": digest_file(
            base / "step_05_classification/training.csv"
        ),
    }
    report(
        root / "step_07_verification/learning_audit",
        "Neural learning-data audit",
        stats,
        {
            "development_query_splits": stats["query_splits"],
            "classification_splits": stats["classification_splits"],
            "class_counts": dict(Counter(r["class_name"] for r in classified)),
        },
    )
    print(json.dumps(stats, indent=2), flush=True)
    return stats
