"""Content classification using reviewed seeds and cosine centroids."""

import csv
import json
from pathlib import Path
import numpy as np
from src.common.artifacts import digest_file, report, write_json
from src.common.config import load_config
from src.common.learning_metrics import classification_metrics
from src.common.zones import ZONES
from src.step_05_classification.labels import load_training_examples


def normalized(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def build_document_vectors(engine):
    """Each available zone contributes one normalized mean, regardless of length."""
    total = np.zeros((len(engine.documents), engine.encoder.dimension), np.float32)
    for zone in ZONES:
        sums = np.zeros_like(total)
        counts = np.zeros(len(total), np.int64)
        for start in range(0, len(engine.matrices[zone]), 8192):
            block = normalized(
                np.asarray(engine.matrices[zone][start : start + 8192], np.float32)
            )
            ids = engine.mappings[zone][start : start + len(block)]
            np.add.at(sums, ids, block)
            np.add.at(counts, ids, 1)
        total += normalized(sums / np.maximum(counts[:, None], 1))
    vectors = normalized(total).astype(np.float32)
    return vectors


def memberships(scores, margin, maximum=2):
    order = np.argsort(-scores, axis=1, kind="stable")
    result = np.full((len(scores), 2), -1, dtype=np.int16)
    result[:, 0] = order[:, 0]
    if maximum == 2:
        use = (
            scores[np.arange(len(scores)), order[:, 0]]
            - scores[np.arange(len(scores)), order[:, 1]]
            <= margin
        )
        result[use, 1] = order[use, 1]
    return result


def train_classifiers(root, engine):
    root = Path(root)
    out = root / "step_05_classification"
    out.mkdir(parents=True, exist_ok=True)
    source = Path("data/step_05_classification/training.csv")
    rows = load_training_examples(root, source)
    with Path("data/step_05_classification/classes.csv").open() as stream:
        definitions = list(csv.DictReader(stream))
    classes = sorted(r["class_id"] for r in definitions)
    by_id = {d["doc_id"]: i for i, d in enumerate(engine.documents)}
    selected = np.array([by_id[r["doc_id"]] for r in rows])
    targets = np.array([classes.index(r["class_id"]) for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    if any(not np.any(train & (targets == c)) for c in range(len(classes))):
        raise ValueError("Every class needs training seeds")
    vectors = build_document_vectors(engine)
    centroids = normalized(
        np.stack(
            [
                vectors[selected[train & (targets == c)]].mean(axis=0)
                for c in range(len(classes))
            ]
        )
    ).astype(np.float32)
    settings = load_config()
    assignments = memberships(
        vectors @ centroids.T, settings["secondary_margin"], settings["max_memberships"]
    )
    np.savez_compressed(
        out / "models.npz", centroids=centroids, document_classes=assignments
    )
    mask = ~train
    predicted = np.argmax(vectors[selected] @ centroids.T, axis=1)
    result = classification_metrics(targets[mask], predicted[mask], len(classes))
    predictions = [
        {"doc_id": r["doc_id"], "actual": r["class_id"], "predicted": classes[int(p)]}
        for r, p, use in zip(rows, predicted, mask)
        if use
    ]
    predictions_path = Path("results/step_05_classification/training/predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["doc_id", "actual", "predicted"])
        writer.writeheader()
        writer.writerows(predictions)
    model = {
        "schema": "content-centroid-v2",
        "index_fingerprint": engine.summary["fingerprint"],
        "classes": classes,
        "class_names": {r["class_id"]: r["class_name"] for r in definitions},
        "training_csv_sha256": digest_file(source),
        "classes_sha256": digest_file("data/step_05_classification/classes.csv"),
        "training_documents": int(train.sum()),
        "validation": result,
        "secondary_margin": settings["secondary_margin"],
        "max_memberships": settings["max_memberships"],
        "document_representation": "Equal available-zone normalized means of normalized windows",
        "label_scope": "Reviewed primary content classes; secondary membership is a similarity heuristic",
        "documents": len(vectors),
        "memberships": int((assignments >= 0).sum()),
    }
    write_json(out / "model.json", model)
    counts = {
        classes[c]: int((assignments == c).any(axis=1).sum())
        for c in range(len(classes))
    }
    report(
        out / "training",
        "Content centroid classification",
        model,
        {
            "class_documents": counts,
            "validation": {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
            },
        },
    )
    print(json.dumps({"classification": model}), flush=True)
    return model


class Classifiers:
    def __init__(self, root):
        root = Path(root)
        directory = root / "step_05_classification"
        self.model = json.loads((directory / "model.json").read_text())
        settings = load_config()
        if (
            self.model.get("schema") != "content-centroid-v2"
            or self.model["index_fingerprint"]
            != json.loads((root / "step_02_windows/summary.json").read_text())[
                "fingerprint"
            ]
        ):
            raise RuntimeError("Run step 05: classifier is stale")
        for key, path in (
            ("training_csv_sha256", "data/step_05_classification/training.csv"),
            ("classes_sha256", "data/step_05_classification/classes.csv"),
        ):
            if self.model[key] != digest_file(path):
                raise RuntimeError(
                    "Run step 05: reviewed classification labels changed"
                )
        if any(
            self.model[k] != settings[k]
            for k in ("secondary_margin", "max_memberships")
        ):
            raise RuntimeError("Run step 05: membership settings changed")
        with np.load(directory / "models.npz") as saved:
            self.centroids = saved["centroids"]
            self.document_classes = saved["document_classes"]

    def query_scores(self, text, vector, method="centroid"):
        if method != "centroid":
            raise ValueError("Only centroid classification is supported")
        return self.centroids @ vector
