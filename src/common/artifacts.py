"""Artifact contracts shared by the independent neural stages."""

from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import sqlite3
from string import Template

from src.common.config import load_config, default_mode

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
    """Persist compact numeric evidence, charts, and a filterable local dashboard."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    write_json(directory / "summary.json", stats)
    try:
        relative = directory.resolve().relative_to(Path("outputs").resolve())
    except ValueError:
        relative = None
    if relative is not None:
        directory = Path("results") / relative
    (directory / "charts").mkdir(parents=True, exist_ok=True)
    write_json(directory / "summary.json", stats)
    if distributions:
        for path in (directory / "charts").glob("*.png"):
            path.unlink()
    figures = []
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
        figures.append(name)
    if not figures:
        figures = [path.stem for path in sorted((directory / "charts").glob("*.png"))]
    (directory / "report.md").write_text(
        "# "
        + title
        + "\n\n```json\n"
        + json.dumps(stats, indent=2)
        + "\n```\n\n"
        + "\n\n".join(f"![{n}](charts/{n}.png)" for n in figures),
        encoding="utf-8",
    )
    body = "".join(
        f'<figure><img style="max-width:100%" src="charts/{n}.png"><figcaption>{html.escape(n)}</figcaption></figure>'
        for n in figures
    )
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in stats.items()
    )
    template = Template(Path(__file__).with_name("report_template.html").read_text())
    (directory / "dashboard.html").write_text(
        template.substitute(title=html.escape(title), rows=rows, figures=body),
        encoding="utf-8",
    )


def save_configuration(root):
    """Record the fixed experiment contract alongside the durable artifacts."""
    root = Path(root)
    write_json(
        root / "config.json",
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "weight_sha256": _SETTINGS["model_weights_sha256"],
            "pooling": "final real token; right padding; omit redundant padding mask for pinned causal Qwen3 decoder",
            "query_instruction": QUERY_INSTRUCTION,
            "max_total_tokens": MAX_TOKENS,
            "zones": ["title", "headings", "body", "code"],
            "max_content_tokens": MAX_TOKENS - _SETTINGS["reserved_tokens"],
            "max_overlap_tokens": OVERLAP,
            "dimensions": json.loads(
                (root / "step_03_embeddings/model/config.json").read_text()
            )["hidden_size"]
            if (root / "step_03_embeddings/model/config.json").exists()
            else None,
            "window_policy": "independent pure-zone windows; no synthetic context prefixes",
            "inference_dtype": "bfloat16",
            "storage_dtype": "float16",
            "scoring_dtype": "float32",
            "ranking": "hand-coded cosine; max window per zone; deterministic doc-ID tie break",
            "retrieval_modes": ["dense", "zones", "routed-centroid"],
            "default_mode": default_mode(_SETTINGS),
            "zone_scoring": "fixed nonnegative learned weighted sum; absent zones contribute zero",
            "routing": "configurable nearest content classes; category row lists restrict scoring",
            "dataset_manifest_sha256": json.loads(
                (root / "step_01_extraction/summary.json").read_text()
            )["dataset_manifest_sha256"],
            "requirements_sha256": digest_file(
                Path(__file__).resolve().parents[2] / "requirements.txt"
            ),
        },
    )
