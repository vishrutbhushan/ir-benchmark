"""Readable source snapshots for documents referenced by the learning tables."""

import csv
import json
from contextlib import closing
from pathlib import Path
import re

from src.common.artifacts import connect, report


def export_reference_documents(root, groups=("classification", "zone_weights")):
    root = Path(root)
    results = {}
    all_ids = set()
    with closing(connect(root / "step_01_extraction/corpus.sqlite", True)) as db:
        for group in groups:
            if group not in {"classification", "zone_weights"}:
                raise ValueError("Unknown learning dataset")
            directory = (
                Path("data")
                / (
                    {
                        "classification": "step_05_classification",
                        "zone_weights": "step_04_weights",
                    }[group]
                )
            )
            with (directory / "training.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                ids = sorted({row["doc_id"] for row in csv.DictReader(stream)})
            destination = directory / "documents"
            destination.mkdir(parents=True, exist_ok=True)
            total_bytes = 0
            for doc_id in ids:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", doc_id):
                    raise ValueError("Invalid reference document ID")
                row = db.execute(
                    "SELECT record_json FROM documents WHERE doc_id=?", (doc_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"Unknown reference document: {doc_id}")
                doc = json.loads(row[0])
                text = (
                    f"Document ID: {doc_id}\nURL: {doc['url']}\n\n"
                    f"TITLE\n{doc['title']}\n\n"
                    "HEADINGS\n" + "\n".join(doc["headings"]) + "\n\n"
                    f"BODY\n{doc['body_text']}\n\nCODE\n{doc['code_text']}\n"
                )
                path = destination / f"{doc_id}.txt"
                temporary = path.with_suffix(".txt.part")
                temporary.write_text(text, encoding="utf-8")
                temporary.replace(path)
                total_bytes += len(text.encode("utf-8"))
            expected = {f"{doc_id}.txt" for doc_id in ids}
            for path in destination.glob("document_*.txt"):
                if path.name not in expected:
                    path.unlink()
            results[group] = {
                "documents": len(ids),
                "bytes": total_bytes,
                "directory": destination.as_posix(),
            }
            all_ids.update(ids)
            print(
                f"{group}: saved {len(ids)} reference documents in {destination}",
                flush=True,
            )
    stats = {
        "datasets": results,
        "unique_documents": len(all_ids),
        "source": "Full extracted source text from SQLite; these files are for reference, not model inputs",
    }
    report(
        root / "step_01_extraction/reference_documents",
        "Learning document references",
        stats,
        {
            "documents": {
                group: values["documents"] for group, values in results.items()
            }
        },
    )
    return stats
