"""Build document row postings for unsupervised cluster-restricted search."""

import json
from pathlib import Path
import numpy as np
from src.common.artifacts import report, write_json
from src.common.config import load_config
from src.common.zones import ZONES


def tune_routing(root, engine):
    root = Path(root)
    out = root / "step_06_routing"
    out.mkdir(parents=True, exist_ok=True)
    clusters = engine.clusters
    arrays, counts = {}, {}
    for cluster in range(clusters.model["clusters"]):
        allowed = clusters.labels == cluster
        counts[str(cluster)] = int(allowed.sum())
        for zone in ZONES:
            arrays[f"{cluster}_{zone}"] = np.flatnonzero(
                allowed[engine.mappings[zone]]
            ).astype(np.int32)
    np.savez_compressed(out / "rows.npz", **arrays)
    model = {
        "schema": "cluster-postings-v1",
        "index_fingerprint": engine.summary["fingerprint"],
        "clusters": clusters.model["clusters"],
        "documents": len(clusters.labels),
        "cluster_documents": counts,
        "default_top_clusters": load_config()["route_clusters"],
        "meaning": "Unsupervised cluster pruning; top-N clusters are searched",
    }
    write_json(out / "model.json", model)
    engine.routing = model
    with np.load(out / "rows.npz") as saved:
        engine.routing_rows = {k: saved[k] for k in saved.files}
    report(out, "Cluster vector row lists", model, {"cluster_documents": counts})
    print(json.dumps(model), flush=True)
    return model
