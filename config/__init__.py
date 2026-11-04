"""Shared repository configuration loading."""

import json
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
    return json.loads(config_path.resolve().read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
