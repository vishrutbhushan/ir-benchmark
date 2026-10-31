"""Read and write compact evaluation files."""

import csv
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "engine",
    "variant",
    "query_id",
    "query_text",
    "rank",
    "doc_id",
    "title",
    "url",
    "technology",
    "categories",
    "score",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(csv_text(rows, columns), encoding="utf-8")
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
