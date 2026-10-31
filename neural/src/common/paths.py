"""Repository-relative paths for neural training inputs."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_INPUTS = REPOSITORY_ROOT / "data_inputs"
NEURAL_WEIGHTS = DATA_INPUTS / "neural_weights"
NEURAL_CLASSIFICATION = DATA_INPUTS / "neural_classification"
