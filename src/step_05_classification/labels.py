"""Load reviewed content labels and resolve source text by document ID."""

from collections import Counter
from contextlib import closing
import csv
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.common.artifacts import connect
from src.common.reference_documents import export_reference_documents


def canonical_family(url):
    parsed = urlsplit(url)
    path = re.sub(r"\d+\.\d+(?:\.\d+)*(?:[-_][\w.-]+)?", "VERSION", parsed.path)
    path = re.sub(r"(?<=/)(?:v?\d+|stable|latest|current)(?=/|$)", "VERSION", path)
    return parsed.netloc.lower().removeprefix("www.") + path.rstrip("/")


def load_training_examples(
    root, source=Path("data/step_05_classification/training.csv")
):
    """Resolve compact labels to the original extracted document text."""
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"doc_id", "class_id", "class_name", "split"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(
                "Classification CSV must contain document IDs, class labels and split"
            )
        rows = list(reader)
    if not rows or len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("Classification documents must be nonempty and unique")
    with closing(connect(Path(root) / "step_01_extraction/corpus.sqlite", True)) as db:
        for row in rows:
            source_row = db.execute(
                "SELECT record_json FROM documents WHERE doc_id=?", (row["doc_id"],)
            ).fetchone()
            if source_row is None:
                raise ValueError(f"Unknown classification document: {row['doc_id']}")
            doc = json.loads(source_row[0])
            row["text"] = "\n\n".join(
                filter(
                    None,
                    [
                        doc["title"],
                        "\n".join(doc["headings"]),
                        doc["body_text"],
                        doc["code_text"],
                    ],
                )
            )
            if not row["text"].strip():
                raise ValueError(f"Empty classification text: {row['doc_id']}")
    return rows


def export_classification_data(root):
    """Validate authoritative reviewed labels; never regenerate them during a run."""
    rows = load_training_examples(root)
    if any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("Only training and validation seed splits are supported")
    export_reference_documents(root, ("classification",))
    return {"documents": len(rows), "splits": dict(Counter(r["split"] for r in rows))}
