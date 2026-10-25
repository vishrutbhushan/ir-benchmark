"""Hand-coded cosine aggregation and fixed-zone weighting."""

import numpy as np


def aggregate_documents(scores, doc_indices, document_count):
    scores = np.asarray(scores, dtype=np.float32)
    doc_indices = np.asarray(doc_indices, dtype=np.int64)
    maxima = np.full(document_count, -np.inf, dtype=np.float32)
    np.maximum.at(maxima, doc_indices, scores)
    winners = np.full(document_count, len(scores), dtype=np.int64)
    tied = np.flatnonzero(scores == maxima[doc_indices])
    np.minimum.at(winners, doc_indices[tied], tied)
    return maxima, winners


def weighted_zone_scores(scores, weights):
    """Fixed global weights; an absent zone contributes zero."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (scores.shape[1],)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Zone weights must be nonnegative and sum to one")
    present = np.isfinite(scores)
    contributions = np.where(present, scores, 0) * weights
    result = contributions.sum(axis=1)
    result[~present.any(axis=1)] = -np.inf
    return result.astype(np.float32), contributions.astype(np.float32)
