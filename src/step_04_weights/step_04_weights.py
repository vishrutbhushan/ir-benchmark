"""Learn four nonnegative zone weights from binary relevance judgments."""

from collections import defaultdict, Counter
import csv
import json
from pathlib import Path

import numpy as np

from src.common.artifacts import digest_file, report, write_json
from src.common.zones import ZONES
from src.step_08_search.scoring import weighted_zone_scores
from src.common.reference_documents import export_reference_documents


def refresh_features(root, engine):
    root = Path(root)
    path = Path("data/step_04_weights/training.csv")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {doc["doc_id"]: i for i, doc in enumerate(engine.documents)}
    groups = defaultdict(list)
    for row in rows:
        if row["doc_id"] not in by_id:
            raise ValueError(f"Labeled document no longer in corpus: {row['doc_id']}")
        groups[row["query_id"]].append(row)
    for qid, group in groups.items():
        if len({(r["query_text"], r["split"]) for r in group}) != 1:
            raise ValueError("Inconsistent query/split")
        values, _, _ = engine.zone_scores(engine._query(group[0]["query_text"]))
        for row in group:
            doc = engine.documents[by_id[row["doc_id"]]]
            row.update(
                doc_title=doc["title"],
                doc_url=doc["url"],
                doc_technology=doc["technology"],
            )
            for column, zone in enumerate(ZONES):
                score = values[by_id[row["doc_id"]], column]
                row[f"has_{zone}"] = str(int(np.isfinite(score)))
                row[f"{zone}_cosine"] = (
                    f"{float(score):.8f}" if np.isfinite(score) else ""
                )
    feature_path = Path("results/step_04_weights/features/training_features.csv")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(feature_path)
    export_reference_documents(root, ("zone_weights",))
    report(
        root / "step_04_weights/features",
        "Full-index zone learning features",
        {
            "pairs": len(rows),
            "queries": len(groups),
            "index_fingerprint": engine.summary["fingerprint"],
            "training_csv_sha256": digest_file(path),
        },
        {"pairs_by_split": dict(Counter(r["split"] for r in rows))},
    )
    print(f"Refreshed {len(rows)} full-index zone feature rows.", flush=True)
    return rows


def fit_weights(root, engine):
    root = Path(root)
    path = Path("data/step_04_weights/training.csv")
    rows = [
        r for r in refresh_features(root, engine) if r["use_for_weight_learning"] == "1"
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    x = np.array(
        [
            [float(r[f"{z}_cosine"]) if r[f"has_{z}"] == "1" else 0.0 for z in ZONES]
            for r in rows
        ]
    )
    y = np.array([int(r["relevant"]) for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    if not train.any() or train.all():
        raise ValueError("Need separate training and validation queries")
    # Squared error is quadratic in the four fixed global weights.
    # Evaluate the simplex grid using its sufficient statistics, not repeated scans.
    xt, yt = x[train], y[train]
    gram, cross, constant = (
        xt.T @ xt / len(xt),
        xt.T @ yt / len(xt),
        float(yt @ yt / len(yt)),
    )
    best = None
    for a in range(101):
        for b in range(101 - a):
            c = np.arange(101 - a - b)
            candidates = (
                np.column_stack(
                    (np.full(len(c), a), np.full(len(c), b), c, 100 - a - b - c)
                )
                / 100
            )
            losses = (
                np.einsum("ij,jk,ik->i", candidates, gram, candidates)
                - 2 * candidates @ cross
                + constant
            )
            for loss, weights in zip(losses, candidates):
                key = (
                    round(float(loss), 12),
                    float(np.sum((weights - 0.25) ** 2)),
                    tuple(weights),
                )
                if best is None or key < best[0]:
                    best = (key, weights.copy())
    weights = best[1]
    scores, _ = weighted_zone_scores(x, weights)
    evaluations = {}
    for split in ("train", "validation"):
        losses = (
            scores[np.array([r["split"] == split for r in rows])]
            - y[np.array([r["split"] == split for r in rows])]
        ) ** 2
        evaluations[split] = {
            "mse": float(losses.mean()),
            "queries": sum(group[0]["split"] == split for group in groups.values()),
        }
    model = {
        "schema": "fixed-zone-sse-v2",
        "zones": list(ZONES),
        "weights": weights.tolist(),
        "index_fingerprint": engine.summary["fingerprint"],
        "training_csv_sha256": digest_file(path),
        "objective": "training-only squared error to binary relevance; deterministic simplex grid step 0.01",
        "missing_zone_policy": "absent zones contribute zero; fixed global weights",
        "evaluation": evaluations,
        "zone_diagnostics": {
            z: {
                "relevant_mean_cosine": float(x[(y == 1) & train, i].mean()),
                "irrelevant_mean_cosine": float(x[(y == 0) & train, i].mean()),
                "training_mse_alone": float(np.mean((x[train, i] - y[train]) ** 2)),
            }
            for i, z in enumerate(ZONES)
        },
        "judgment_scope": "Small judged development pools, not final retrieval evaluation",
    }
    write_json(root / "step_04_weights/model.json", model)
    report(
        root / "step_04_weights",
        "Learned four-zone weights",
        model,
        {
            "weights": dict(zip(ZONES, weights.tolist())),
            "validation": {
                k: v for k, v in evaluations["validation"].items() if k != "queries"
            },
        },
    )
    engine.weights = weights.astype(np.float32)
    print(json.dumps(model, indent=2), flush=True)
    return model
