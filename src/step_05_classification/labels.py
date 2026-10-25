"""Read content labels and check document IDs."""

from contextlib import closing
import csv
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.common.artifacts import connect


def canonical_family(url):
    parsed = urlsplit(url)
    path = re.sub(r"\d+\.\d+(?:\.\d+)*(?:[-_][\w.-]+)?", "VERSION", parsed.path)
    path = re.sub(r"(?<=/)(?:v?\d+|stable|latest|current)(?=/|$)", "VERSION", path)
    return parsed.netloc.lower().removeprefix("www.") + path.rstrip("/")


def load_training_examples(
    root, source=Path("data/step_05_classification/training.csv")
):
    """Validate seed labels without copying document text."""
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
            if row["split"] not in {"train", "validation"}:
                raise ValueError("Seed split must be train or validation")
            source_row = db.execute(
                "SELECT 1 FROM documents WHERE doc_id=?", (row["doc_id"],)
            ).fetchone()
            if source_row is None:
                raise ValueError(f"Unknown classification document: {row['doc_id']}")
    return rows
