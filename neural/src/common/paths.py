"""Repository-relative paths for neural training inputs."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NEURAL_DATA = REPOSITORY_ROOT / "neural/data"
