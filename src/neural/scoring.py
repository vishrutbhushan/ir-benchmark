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
    """Renormalize nonnegative weights over present zones; fall back if all are zero."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (scores.shape[1],) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Zone weights must be nonnegative with positive total')
    present = np.isfinite(scores)
    effective = present * weights
    denominator = effective.sum(axis=1)
    fallback = (denominator == 0) & present.any(axis=1)
    effective[fallback] = present[fallback]
    denominator = effective.sum(axis=1)
    contributions = np.where(present, scores, 0) * effective / np.maximum(denominator[:, None], 1e-15)
    result = contributions.sum(axis=1)
    result[~present.any(axis=1)] = -np.inf
    return result.astype(np.float32), contributions.astype(np.float32)
