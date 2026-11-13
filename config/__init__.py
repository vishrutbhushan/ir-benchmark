"""Shared repository configuration loading."""

import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_pipeline_config(path: str | Path | None = None) -> dict:
    config_path = Path(
        path or os.environ.get("IR_CONFIG", ROOT / "config/pipeline.json")
    )
    if not config_path.is_absolute():
        config_path = (
            config_path
            if config_path.exists()
            else ROOT / config_path
        )
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    classical = config["classical"]
    zones = {"title", "heading", "body", "code"}
    for key in ("zone_weights", "zone_b"):
        values = classical[key]
        if set(values) != zones:
            raise ValueError(f"classical.{key} must define exactly {sorted(zones)}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values.values()
        ):
            raise ValueError(f"classical.{key} values must be nonnegative numbers")
    if any(value > 1 for value in classical["zone_b"].values()):
        raise ValueError("classical.zone_b values must be at most one")
    rrf = config["evaluation"]["rrf"]
    if not isinstance(rrf["k"], int) or isinstance(rrf["k"], bool) or rrf["k"] <= 0:
        raise ValueError("evaluation.rrf.k must be a positive integer")
    weights = (rrf["bm25f_weight"], rrf["routed_weight"])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in weights
    ) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("evaluation RRF weights must be nonnegative and sum to one")
    expected_runs = {
        ("classical", "bm25"),
        ("classical", "bm25f"),
        ("neural", "dense"),
        ("neural", "zones"),
        ("neural", "routed"),
        ("hybrid", "rrf"),
    }
    snapshots = config["evaluation"]["result_snapshots"]
    configured_runs = {(item["engine"], item["variant"]) for item in snapshots}
    if configured_runs != expected_runs or len(snapshots) != len(expected_runs):
        raise ValueError("evaluation.result_snapshots must define the six active systems")
    search_engines = config["evaluation"]["search_engines"]
    if len(search_engines) != 6 or len({item["id"] for item in search_engines}) != 6:
        raise ValueError("evaluation.search_engines must define six unique UI choices")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
