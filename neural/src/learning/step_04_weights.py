"""Learn reproducible global zone weights using Manning Chapter 6 weighted zone scoring."""

import csv
import json
from pathlib import Path
import numpy as np

from src.common.artifacts import digest_file, write_json
from src.common.paths import REPOSITORY_ROOT
from src.common.config import load_config
from src.common.zones import ZONES


def project_simplex(v, minimum=0.0):
    """Project onto a probability simplex with an optional minimum floor per component."""
    k = len(v)
    residual = 1.0 - k * minimum
    if residual <= 0:
        return np.full(k, 1.0 / k, dtype=np.float64)
    shifted = np.asarray(v, dtype=np.float64) - minimum
    u = np.sort(shifted)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, k + 1) > (cssv - residual))[0][-1]
    theta = (cssv[rho] - residual) / (rho + 1)
    projected = np.maximum(shifted - theta, 0.0) + minimum
    return projected / np.sum(projected)


def fit_manning_weights(X, y, minimum=0.05):
    """
    Solve constrained least-squares:
    min ||X w - y||^2 subject to sum(w) = 1, w_i >= minimum.
    Pure NumPy implementation of projected gradient descent.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.full(X.shape[1], 1.0 / X.shape[1], dtype=np.float64)
    
    XtX = X.T @ X
    Xty = X.T @ y
    eigenvalues = np.linalg.eigvalsh(XtX)
    L = float(np.max(eigenvalues)) if float(np.max(eigenvalues)) > 1e-6 else 1.0
    step = 1.0 / L

    for _ in range(1000):
        grad = XtX @ w - Xty
        w_next = project_simplex(w - step * grad, minimum=minimum)
        if np.max(np.abs(w_next - w)) < 1e-12:
            break
        w = w_next
    return w


def configure_weights(root, engine=None):
    if engine is None:
        raise RuntimeError(
            "Weight learning requires an initialized neural index; run --step all "
            "or --step 04 after stages 01-03 with the same config."
        )
    root = Path(root)
    settings = load_config()
    judgments_path = REPOSITORY_ROOT / settings.get(
        "weight_judgments", "data_inputs/neural_weights/manning_zone_judgments.csv"
    )
    if not judgments_path.exists():
        judgments_path = REPOSITORY_ROOT / "data_inputs/neural_weights/manning_zone_judgments.csv"

    with open(judgments_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X_list = []
    y_list = []
    for r in rows:
        # Zones order: title, headings, body, code
        s_T = float(r["s_T"])
        s_H = float(r["s_H"])
        s_B = float(r["s_B"])
        s_C = float(r["s_C"])
        rel = float(r["relevant_r"])
        X_list.append([s_T, s_H, s_B, s_C])
        y_list.append(rel)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    # Solve unconstrained and constrained least-squares
    unrestricted, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Fit Manning weights with minimum 5% floor
    minimum_floor = float(settings.get("minimum_zone_weight", 0.05))
    weights = fit_manning_weights(X, y, minimum=minimum_floor)

    # Calculate correlation with relevance
    separation = {}
    for idx, zone in enumerate(ZONES):
        col = X[:, idx]
        corr = float(np.corrcoef(col, y)[0, 1]) if np.std(col) > 0 and np.std(y) > 0 else 0.0
        separation[zone] = {
            "correlation_with_relevance": corr,
            "mean_relevant": float(np.mean(col[y == 1])) if np.sum(y == 1) > 0 else 0.0,
            "mean_non_relevant": float(np.mean(col[y == 0])) if np.sum(y == 0) > 0 else 0.0,
        }

    configured = np.asarray([settings["zone_weights"][z] for z in ZONES], dtype=float)
    predictions = X @ weights
    mse = float(np.mean((predictions - y) ** 2))

    model = {
        "schema": "query-balanced-zone-ridge-v1",
        "method": "Manning Chapter 6 Section 6.1 Weighted Zone Scoring",
        "zones": list(ZONES),
        "weights": weights.tolist(),
        "unrestricted_least_squares_weights": unrestricted.tolist(),
        "configured_weights": configured.tolist(),
        "index_fingerprint": engine.summary["fingerprint"],
        "judgments_csv": str(judgments_path.relative_to(REPOSITORY_ROOT)),
        "judgments_csv_sha256": digest_file(judgments_path),
        "minimum_zone_weight": minimum_floor,
        "examples": len(rows),
        "mse": mse,
        "per_zone_correlation": separation,
    }

    out_dir = root / "step_04_weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "model.json", model)
    engine.weights = weights.astype(np.float32)

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
                "learned_weight": f"{w:.4f}",
                "configured_weight": f"{c:.4f}",
                "minimum_weight": f"{minimum_floor:.4f}",
            }
            for zone, w, c in zip(ZONES, weights, configured)
        )

    print(json.dumps(model, indent=2), flush=True)
    return model
