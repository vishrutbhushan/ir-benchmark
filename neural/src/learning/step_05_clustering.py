"""Unsupervised document clustering over normalized title and headings representations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.artifacts import report, write_json
from src.common.config import load_config
from src.common.zones import ZONES

CLUSTERING_ZONES = ("title", "headings")


def build_zone_vectors(engine):
    """Return one title and headings vector per document and presence masks."""
    return engine.zone_document_vectors()


def _features(vectors, present, weights):
    if isinstance(weights, dict):
        raw_weights = [weights.get(z, 0.5) for z in CLUSTERING_ZONES]
    else:
        raw_weights = [0.5, 0.5]
    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]
    scaled = vectors * np.sqrt(np.asarray(norm_weights, dtype=np.float32))[None, :, None]
    mask = present.astype(np.float32) * 0.1
    feats = np.concatenate((scaled.reshape(len(vectors), -1), mask), axis=1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-12)


def _kmeans_pp_init(features, k, rng):
    n = len(features)
    centers = np.empty((k, features.shape[1]), dtype=features.dtype)
    first_idx = rng.integers(n)
    centers[0] = features[first_idx]
    dist_sq = np.sum((features - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = dist_sq.sum()
        probs = dist_sq / total if total > 1e-12 else np.full(n, 1.0 / n)
        next_idx = rng.choice(n, p=probs)
        centers[i] = features[next_idx]
        new_dist = np.sum((features - centers[i]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist)
    return centers


def _kmeans(features, k, seed, iterations=100, n_init=3):
    if k > len(features):
        raise ValueError("cluster_k cannot exceed the number of indexed documents")
    best_labels, best_centers, best_inertia = None, None, float("inf")
    base_rng = np.random.default_rng(seed)

    for _ in range(n_init):
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        centers = _kmeans_pp_init(features, k, rng)
        for _ in range(iterations):
            distances = (
                np.sum(features * features, axis=1)[:, None]
                + np.sum(centers * centers, axis=1)[None, :]
                - 2 * (features @ centers.T)
            )
            labels = np.argmin(distances, axis=1)
            updated = centers.copy()
            for cluster in range(k):
                members = features[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = features[rng.integers(len(features))]
            if np.allclose(updated, centers, atol=1e-6):
                centers = updated
                break
            centers = updated
        distances = np.sum((features - centers[labels]) ** 2, axis=1)
        inertia = float(distances.sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_inertia


def train_clusters(root, engine):
    root = Path(root)
    out = root / "step_05_clustering"
    out.mkdir(parents=True, exist_ok=True)
    settings = load_config()
    vectors, present = build_zone_vectors(engine)
    if engine.weights is not None:
        weights = dict(zip(ZONES, map(float, engine.weights)))
    else:
        weights = settings["zone_weights"]
    features = _features(vectors, present, weights)
    k = min(settings["cluster_k"], len(features))
    labels, centers, inertia = _kmeans(
        features, k, settings["cluster_seed"], settings["cluster_iterations"]
    )

    # Multi-membership assignment with secondary_margin
    c_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    c_normed = centers / np.maximum(c_norms, 1e-12)
    sims = features @ c_normed.T
    sorted_clusters = np.argsort(-sims, axis=1)
    primary = sorted_clusters[:, 0]
    max_sims = np.max(sims, axis=1)

    secondary_margin = float(settings.get("secondary_margin", 0.03))
    second_cluster = sorted_clusters[:, 1]
    second_sims = sims[np.arange(len(sims)), second_cluster]
    has_secondary = second_sims >= (max_sims - secondary_margin)

    memberships = np.full((len(labels), 2), -1, dtype=np.int32)
    memberships[:, 0] = primary
    memberships[has_secondary, 1] = second_cluster[has_secondary]

    np.savez_compressed(
        out / "zone_vectors.npz",
        vectors=vectors.astype(np.float16),
        present=present,
    )
    np.savez_compressed(
        out / "model.npz",
        labels=primary.astype(np.int32),
        memberships=memberships.astype(np.int32),
        centers=centers.astype(np.float32),
    )
    counts = {str(c): int((primary == c).sum()) for c in range(k)}
    multi_counts = {str(c): int((np.any(memberships == c, axis=1)).sum()) for c in range(k)}
    raw_weights = [weights.get(z, 0.5) if isinstance(weights, dict) else 0.5 for z in CLUSTERING_ZONES]
    total_w = sum(raw_weights)
    model = {
        "schema": "unsupervised-zone-kmeans-v1",
        "index_fingerprint": engine.summary["fingerprint"],
        "clusters": k,
        "seed": settings["cluster_seed"],
        "iterations": settings["cluster_iterations"],
        "inertia": inertia,
        "cluster_documents": counts,
        "multi_membership_documents": multi_counts,
        "secondary_margin": secondary_margin,
        "secondary_assignments": int(has_secondary.sum()),
        "representation": "Title and Headings embeddings from Qwen3-Embedding-0.6B with k-means++ initialization",
        "zone_embedding_dimension": int(vectors.shape[2]),
        "zone_vectors": "zone_vectors.npz",
        "zones": list(CLUSTERING_ZONES),
        "documents": len(labels),
        "zone_weights": {
            zone: float(w / total_w) for zone, w in zip(CLUSTERING_ZONES, raw_weights)
        },
    }
    write_json(out / "model.json", model)
    report(out, "Unsupervised document clustering", model, {"cluster_documents": counts})
    return model


class Clusters:
    def __init__(self, root):
        root = Path(root)
        directory = root / "step_05_clustering"
        self.model = json.loads((directory / "model.json").read_text())
        summary = json.loads((root / "step_02_windows/summary.json").read_text())
        if (
            self.model.get("schema") != "unsupervised-zone-kmeans-v1"
            or self.model["index_fingerprint"] != summary["fingerprint"]
        ):
            raise RuntimeError("Run step 05: clustering is stale")
        with np.load(directory / "model.npz") as saved:
            self.labels = saved["labels"]
            self.centers = saved["centers"]
            self.memberships = (
                saved["memberships"]
                if "memberships" in saved
                else np.column_stack((self.labels, np.full_like(self.labels, -1)))
            )
            if "zone_vectors" not in self.model:
                self.zone_vectors = saved["zone_vectors"].astype(np.float32)
                self.zone_present = saved["zone_present"]
            else:
                self.zone_vectors = None
                self.zone_present = None
        if self.model.get("zone_vectors"):
            with np.load(directory / self.model["zone_vectors"]) as saved:
                self.zone_vectors = saved["vectors"].astype(np.float32)
                self.zone_present = saved["present"]
        if len(self.labels) != self.model["documents"]:
            raise RuntimeError("Cluster assignment count does not match index")

    def query_scores(self, vector):
        dimension = vector.shape[0]
        raw_weights = [self.model["zone_weights"].get(z, 0.5) for z in CLUSTERING_ZONES]
        total_w = sum(raw_weights)
        norm_weights = [np.sqrt(w / total_w) for w in raw_weights]
        q_zones = np.concatenate([vector * w for w in norm_weights])
        q_feat = np.concatenate([q_zones, np.array([0.1, 0.1], dtype=np.float32)])
        q_feat /= np.maximum(np.linalg.norm(q_feat), 1e-12)

        c_norms = np.linalg.norm(self.centers, axis=1, keepdims=True)
        c_normed = self.centers / np.maximum(c_norms, 1e-12)
        return (c_normed @ q_feat).astype(np.float32)
