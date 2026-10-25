"""Load manual zone weights without changing previous learning results."""

import json
from pathlib import Path
import numpy as np
from src.common.artifacts import write_json
from src.common.config import load_config
from src.common.zones import ZONES


def configure_weights(root, engine=None):
    root = Path(root)
    summary = (
        engine.summary
        if engine is not None
        else json.loads((root / "step_02_windows/summary.json").read_text())
    )
    settings = load_config()
    weights = [settings["zone_weights"][zone] for zone in ZONES]
    model = {
        "schema": "manual-zone-weights-v1",
        "source": "config.json zone_weights",
        "zones": list(ZONES),
        "weights": weights,
        "index_fingerprint": summary["fingerprint"],
        "missing_zone_policy": "Absent zones contribute zero; fixed global weights",
    }
    write_json(Path(root) / "step_04_weights/model.json", model)
    if engine is not None:
        engine.weights = np.asarray(weights, dtype=np.float32)
    print(json.dumps(model, indent=2), flush=True)
    return model
