"""Validated project settings. Explicit command-line options override run defaults."""

import json
import os
import math

from src.common.zones import ZONES
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_pipeline_config():
    configured = os.environ.get("NEURAL_CONFIG")
    path = Path(configured) if configured else REPOSITORY_ROOT / "config/pipeline.json"
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def load_config():
    pipeline = load_pipeline_config()
    values = pipeline.get("neural", pipeline)
    for key in (
        "max_tokens",
        "reserved_tokens",
        "overlap",
        "batch_size",
        "workers",
        "max_memberships",
        "query_cache_size",
        "ranking_cache_size",
        "result_limit",
        "page_size",
        "cluster_k",
        "cluster_seed",
        "cluster_iterations",
        "route_clusters",
    ):
        if type(values[key]) is not int or values[key] < (
            0 if key in {"reserved_tokens", "overlap"} else 1
        ):
            raise ValueError(f"Invalid {key}")
    weights = values["zone_weights"]
    if set(weights) != set(ZONES) or any(
        type(w) not in {int, float} or not math.isfinite(w) or w <= 0
        for w in weights.values()
    ):
        raise ValueError(
            "zone_weights must specify a positive number for each of the four zones"
        )
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("zone_weights must sum to one")
    if not 0 < values["minimum_zone_weight"] < 1 / len(ZONES):
        raise ValueError("minimum_zone_weight must be positive and below uniform weight")
    budget = values["max_tokens"] - values["reserved_tokens"]
    if (
        budget < 4
        or not 0 <= values["overlap"] < budget
        or values["overlap"] > budget // 4
    ):
        raise ValueError("Overlap must fit within one quarter of the content window")
    if values["device"] not in {"cpu", "cuda"} or values["max_memberships"] not in {1, 2}:
        raise ValueError("Unsupported device or membership count")
    if values["cluster_k"] < 2 or values["cluster_iterations"] < 1:
        raise ValueError("Clustering requires at least two clusters and one iteration")
    if not 1 <= values["route_clusters"] <= values["cluster_k"]:
        raise ValueError("route_clusters must be within cluster_k")
    if not 0 <= values["secondary_margin"] <= 2:
        raise ValueError("Invalid secondary margin")
    for key in ("zone_weighting", "use_learned_zone_weights", "cluster_routing"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    if not values["model_id"].startswith("Qwen/Qwen3-Embedding-"):
        raise ValueError(
            "Only Qwen3 embedding models use the implemented pooling contract"
        )
    if len(values["model_revision"]) != 40 or len(values["model_weights_sha256"]) != 64:
        raise ValueError("Pin a model commit and model.safetensors SHA-256")
    return values


def default_mode(config):
    return "routed-cluster" if config["cluster_routing"] else (
        "zones" if config["zone_weighting"] else "dense"
    )
