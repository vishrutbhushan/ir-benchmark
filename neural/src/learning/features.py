"""Read and validate the focused zone-weight judgment set."""

import csv
from pathlib import Path


def load_examples(queries_path, judgments_path):
    with Path(queries_path).open(newline="", encoding="utf-8-sig") as stream:
        queries = list(csv.DictReader(stream))
    with Path(judgments_path).open(newline="", encoding="utf-8-sig") as stream:
        judgments = list(csv.DictReader(stream))
    required_queries = {"query_id", "query_text", "focus_zone", "split"}
    required_judgments = {
        "query_id",
        "doc_id",
        "relevance_grade",
        "evidence_zone",
        "evidence_text",
        "rationale",
    }
    if set(queries[0] if queries else ()) != required_queries:
        raise ValueError("Zone-weight query CSV has an invalid schema")
    if set(judgments[0] if judgments else ()) != required_judgments:
        raise ValueError("Zone-weight judgment CSV has an invalid schema")
    query_map = {}
    for row in queries:
        if row["query_id"] in query_map or row["focus_zone"] not in {
            "title",
            "headings",
            "body",
            "code",
        }:
            raise ValueError("Zone-weight queries must have unique IDs and valid zones")
        if row["split"] not in {"train", "validation"} or not row["query_text"].strip():
            raise ValueError("Zone-weight query has invalid text or split")
        query_map[row["query_id"]] = row
    if not query_map or not judgments:
        raise ValueError("Zone-weight CSVs must not be empty")
    seen = set()
    for row in judgments:
        key = (row["query_id"], row["doc_id"])
        if key in seen or row["query_id"] not in query_map:
            raise ValueError("Zone-weight judgments contain an unknown or duplicate pair")
        seen.add(key)
        grade = float(row["relevance_grade"])
        if grade not in {0.0, 1.0} or row["evidence_zone"] not in {
            "title",
            "headings",
            "body",
            "code",
        }:
            raise ValueError("Zone-weight judgment has invalid grade or evidence zone")
        if not row["evidence_text"].strip() or not row["rationale"].strip():
            raise ValueError("Zone-weight judgment evidence and rationale are required")
    return [
        {
            **row,
            **{
                "query_text": query_map[row["query_id"]]["query_text"],
                "split": query_map[row["query_id"]]["split"],
                "focus_zone": query_map[row["query_id"]]["focus_zone"],
            },
        }
        for row in judgments
    ], queries
