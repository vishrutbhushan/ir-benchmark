"""Read the self-contained query/document relevance table."""

import csv
from pathlib import Path


def load_examples(dataset):
    """One self-contained training table; no joins or evidence dependency."""
    with (Path(dataset) / "training.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training table contains no examples")
    splits = {}
    for row in rows:
        signature = (row["query_text"], row["split"])
        if signature != splits.setdefault(row["query_id"], signature):
            raise ValueError("A query ID must have one query text and one split")
        if row["use_for_weight_learning"] not in {"0", "1"}:
            raise ValueError("Weight-learning eligibility must be binary")
        if row["relevant"] not in {"0", "1"}:
            raise ValueError("Relevance must be binary")
    return rows
