"""Learn reproducible global zone weights from reviewed relevance examples."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.artifacts import digest_file, write_json
from src.common.paths import REPOSITORY_ROOT
from src.common.config import load_config
from src.common.zones import ZONES
from src.learning.features import load_examples


def _configured_input(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _project_with_floor(values, minimum):
    """Project onto a simplex while guaranteeing every zone has a floor."""
    if not 0 <= minimum < 1 / len(values):
        raise ValueError("minimum zone weight must be below the uniform weight")
    residual = 1 - minimum * len(values)
    shifted = np.asarray(values, dtype=np.float64) - minimum
    shifted = np.maximum(shifted, 0)
    if shifted.sum() == 0:
        return np.full(len(values), 1 / len(values), dtype=np.float64)
    shifted *= residual / shifted.sum()
    return shifted + minimum


def _fit_weights(features, labels, query_ids, minimum, regularization):
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    counts = defaultdict(int)
    for query_id in query_ids:
        counts[query_id] += 1
    sample_weights = np.asarray(
        [1 / counts[query_id] for query_id in query_ids], dtype=np.float64
    )
    sample_weights /= sample_weights.sum()
    weights = np.full(matrix.shape[1], 1 / matrix.shape[1], dtype=np.float64)
    center = weights.copy()
    for _ in range(2500):
        residual = matrix @ weights - target
        gradient = 2 * (
            matrix.T @ (sample_weights * residual)
            + regularization * (weights - center)
        )
        updated = _project_with_floor(weights - 0.08 * gradient, minimum)
        if np.max(np.abs(updated - weights)) < 1e-10:
            break
        weights = updated
    return weights


def _write_features(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "doc_id",
        "query_text",
        "relevance_grade",
        "split",
        "focus_zone",
        *sum(([f"has_{z}", f"{z}_cosine"] for z in ZONES), []),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_weights(root, engine=None):
    if engine is None:
        raise RuntimeError(
            "Weight learning requires an initialized neural index; run --step all "
            "or --step 04 after stages 01-03 with the same config."
        )
    root = Path(root)
    settings = load_config()
    queries_path = _configured_input(
        settings.get("weight_queries", "data_inputs/neural_weights/zone_weight_queries.csv")
    )
    judgments_path = _configured_input(
        settings.get(
            "weight_judgments",
            "data_inputs/neural_weights/zone_weight_judgments.csv",
        )
    )
    labels, queries = load_examples(queries_path, judgments_path)
    doc_indices = {doc["doc_id"]: i for i, doc in enumerate(engine.documents)}
    grouped = defaultdict(list)
    for row in labels:
        grouped[row["query_id"]].append(row)

    feature_rows = []
    feature_matrix, targets, query_ids, splits = [], [], [], []
    for query_id, query_rows in grouped.items():
        query = query_rows[0]["query_text"]
        scores, _, _ = engine.zone_scores(engine._query(query))
        for row in query_rows:
            if row["doc_id"] not in doc_indices:
                raise ValueError(f"Unknown weight-learning document: {row['doc_id']}")
            index = doc_indices[row["doc_id"]]
            vector, output = [], {
                "query_id": query_id,
                "doc_id": row["doc_id"],
                "query_text": query,
                "relevance_grade": row["relevance_grade"],
                "split": row["split"],
                "focus_zone": row["focus_zone"],
            }
            for column, zone in enumerate(ZONES):
                score = float(scores[index, column])
                present = int(np.isfinite(score))
                output[f"has_{zone}"] = str(present)
                output[f"{zone}_cosine"] = f"{score:.9f}" if present else ""
                vector.append(score if present else 0.0)
            feature_rows.append(output)
            if row["split"] in {"train", "validation"}:
                feature_matrix.append(vector)
                targets.append(float(row["relevance_grade"]))
                query_ids.append(query_id)
                splits.append(row["split"])

    features_path = REPOSITORY_ROOT / "neural/results/step_04_weights/features/training_features.csv"
    _write_features(features_path, feature_rows)
    minimum = float(settings.get("minimum_zone_weight", 0.01))
    regularization = float(settings.get("weight_regularization", 0.05))
    train_mask = [split == "train" for split in splits]
    train_matrix = np.asarray(feature_matrix)[train_mask]
    train_target = np.asarray(targets)[train_mask]
    unrestricted, _, _, _ = np.linalg.lstsq(train_matrix, train_target, rcond=None)
    weights = _fit_weights(
        train_matrix,
        train_target,
        [query_id for query_id, selected in zip(query_ids, train_mask) if selected],
        minimum,
        regularization,
    )
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    predictions = matrix @ weights
    metrics = {}
    for split in ("train", "validation"):
        selected = np.asarray([value == split for value in splits])
        metrics[split] = {
            "examples": int(selected.sum()),
            "mse": float(np.mean((predictions[selected] - target[selected]) ** 2)),
            "mean_prediction_high_grade": float(
                np.mean(predictions[selected & (target >= 0.75)])
            ),
            "mean_prediction_low_grade": float(
                np.mean(predictions[selected & (target <= 0.25)])
            ),
        }
    separation = {}
    for column, zone in enumerate(ZONES):
        relevant = matrix[target >= 0.75, column]
        non_relevant = matrix[target <= 0.25, column]
        separation[zone] = {
            "relevant_mean": float(np.mean(relevant)),
            "non_relevant_mean": float(np.mean(non_relevant)),
            "difference": float(np.mean(relevant) - np.mean(non_relevant)),
        }
    configured = np.asarray([settings["zone_weights"][z] for z in ZONES], dtype=float)
    model = {
        "schema": "query-balanced-zone-ridge-v1",
        "zones": list(ZONES),
        "weights": weights.tolist(),
        "unrestricted_least_squares_weights": unrestricted.tolist(),
        "configured_weights": configured.tolist(),
        "index_fingerprint": engine.summary["fingerprint"],
        "queries_csv_sha256": digest_file(queries_path),
        "judgments_csv_sha256": digest_file(judgments_path),
        "features_csv": str(features_path.relative_to(REPOSITORY_ROOT)),
        "objective": "Query-balanced squared binary relevance error with uniform-weight regularization",
        "minimum_zone_weight": minimum,
        "weight_regularization": regularization,
        "examples": len(feature_rows),
        "eligible_examples": len(labels),
        "eligible_queries": len(queries),
        "relevance_labels": [0, 1],
        "missing_zone_policy": "Absent zones contribute zero; learned global weights",
        "metrics": metrics,
        "per_zone_separation": separation,
    }
    weights_csv = REPOSITORY_ROOT / "neural/results/step_04_weights/weights.csv"
    weights_csv.parent.mkdir(parents=True, exist_ok=True)
    with weights_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["zone", "learned_weight", "configured_weight", "minimum_weight"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "zone": zone,
                "learned_weight": f"{learned:.9f}",
                "configured_weight": f"{configured_value:.9f}",
                "minimum_weight": f"{minimum:.9f}",
            }
            for zone, learned, configured_value in zip(ZONES, weights, configured)
        )
    model["weights_csv"] = str(weights_csv.relative_to(REPOSITORY_ROOT))
    write_json(root / "step_04_weights/model.json", model)
    engine.weights = weights.astype(np.float32)
    print(json.dumps(model, indent=2), flush=True)
    return model
