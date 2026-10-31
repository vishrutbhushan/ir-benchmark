"""Read metadata, write reports and manage SQLite connections."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from src.common.config import load_config

_SETTINGS = load_config()
MODEL_ID = _SETTINGS["model_id"]
MODEL_REVISION = _SETTINGS["model_revision"]
QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
MAX_TOKENS = _SETTINGS["max_tokens"]
OVERLAP = _SETTINGS["overlap"]


def records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def connect(path, readonly=False):
    if readonly:
        db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    return db


def report(directory, title, stats, distributions):
    """Save stage metadata and learning charts."""
    directory = Path(directory)
    try:
        relative = directory.resolve().relative_to(Path("outputs").resolve())
    except ValueError:
        relative = None
    if relative is not None:
        if relative.parts[0] not in {
            "step_04_weights",
            "step_05_classification",
            "step_06_routing",
            "step_07_verification",
        }:
            write_json(directory / "summary.json", stats)
            return
        directory = Path("results") / relative
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (directory / "charts").mkdir(parents=True, exist_ok=True)
    write_json(directory / "summary.json", stats)
    if distributions:
        for path in (directory / "charts").glob("*.png"):
            path.unlink()
    for name, values in distributions.items():
        if not values:
            continue
        labels, counts = zip(*values.items())
        fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.26)))
        ax.barh([str(x) for x in labels], counts, color="#276b80")
        ax.set_xlabel("Count / measured value")
        ax.set_title(name.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(directory / "charts" / f"{name}.png", dpi=125)
        plt.close(fig)
