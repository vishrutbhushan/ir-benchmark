"""Hand-coded classification accuracy, macro-F1 and confusion matrix."""

import numpy as np


def classification_metrics(expected, predicted, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, prediction in zip(expected, predicted):
        matrix[int(actual), int(prediction)] += 1
    f1 = []
    for index in range(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index].sum()) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_f1": sum(f1) / classes,
        "per_class_f1": f1,
        "confusion": matrix.tolist(),
        "documents": int(matrix.sum()),
    }
