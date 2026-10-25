"""Validated project settings. Explicit command-line options override run defaults."""

import json
import os
from pathlib import Path


def load_config():
    path = Path(os.environ.get("NEURAL_CONFIG", "config.json"))
    values = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "max_tokens",
        "reserved_tokens",
        "overlap",
        "batch_size",
        "workers",
        "route_classes",
        "max_memberships",
        "query_cache_size",
        "ranking_cache_size",
        "result_limit",
        "page_size",
    ):
        if type(values[key]) is not int or values[key] < (
            0 if key in {"reserved_tokens", "overlap"} else 1
        ):
            raise ValueError(f"Invalid {key}")
    budget = values["max_tokens"] - values["reserved_tokens"]
    if (
        budget < 4
        or not 0 <= values["overlap"] < budget
        or values["overlap"] > budget // 4
    ):
        raise ValueError("Overlap must fit within one quarter of the content window")
    if values["device"] not in {"cpu", "cuda"} or values["max_memberships"] not in {
        1,
        2,
    }:
        raise ValueError("Unsupported device or membership count")
    if not 0 <= values["secondary_margin"] <= 2:
        raise ValueError("Invalid secondary margin")
    for key in ("zone_weighting", "category_routing"):
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
    return (
        "routed-centroid"
        if config["category_routing"]
        else ("zones" if config["zone_weighting"] else "dense")
    )
