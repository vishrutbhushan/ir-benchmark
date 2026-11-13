"""Resolve repository paths from the shared pipeline configuration."""

from pathlib import Path
from src.common.config import load_pipeline_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def configured_path(key):
    value = Path(load_pipeline_config()["paths"][key])
    return value if value.is_absolute() else REPOSITORY_ROOT / value
