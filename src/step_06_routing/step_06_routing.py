"""Build category row lists over existing embeddings, without duplicating vectors."""

import json
from pathlib import Path
import numpy as np
from src.common.artifacts import digest_file, report, write_json
from src.common.config import load_config
from src.common.zones import ZONES


def tune_routing(root, engine):
    root = Path(root)
    out = root / "step_06_routing"
    out.mkdir(parents=True, exist_ok=True)
    memberships = engine.classifier.document_classes
    arrays, counts = {}, {}
    for c, cid in enumerate(engine.classifier.model["classes"]):
        allowed = (memberships == c).any(axis=1)
        counts[cid] = int(allowed.sum())
        for zone in ZONES:
            arrays[f"{c}_{zone}"] = np.flatnonzero(
                allowed[engine.mappings[zone]]
            ).astype(np.int32)
    np.savez_compressed(out / "rows.npz", **arrays)
    model = {
        "schema": "category-postings-v2",
        "index_fingerprint": engine.summary["fingerprint"],
        "classifier_sha256": digest_file(root / "step_05_classification/model.json"),
        "default_top_classes": load_config()["route_classes"],
        "documents": len(memberships),
        "class_documents": counts,
        "meaning": "Configurable category pruning; no relevance-retention guarantee",
    }
    write_json(out / "model.json", model)
    engine.routing = model
    with np.load(out / "rows.npz") as saved:
        engine.routing_rows = {k: saved[k] for k in saved.files}
    report(out, "Category vector row lists", model, {"class_documents": counts})
    print(json.dumps(model), flush=True)
    return model
