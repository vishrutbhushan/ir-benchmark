"""Textbook ranking and classification metrics, without metric libraries."""
import math
import numpy as np


def dcg(grades, k=5):
    return sum((2**grade-1)/math.log2(position+2) for position,grade in enumerate(grades[:k]))


def ndcg(ranking, judgments, k=5):
    ideal = dcg(sorted(judgments.values(), reverse=True), k)
    return dcg([judgments.get(doc_id, 0) for doc_id in ranking], k)/ideal if ideal else 0.0


def average_precision(ranking, judgments):
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    if not relevant: return 0.0
    found, total = 0, 0.0
    seen = set()
    for position, doc_id in enumerate(ranking, 1):
        if doc_id in seen: raise ValueError('A ranking contains duplicate documents')
        seen.add(doc_id)
        if doc_id in relevant:
            found += 1
            total += found/position
    return total/len(relevant)


def classification_metrics(expected, predicted, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, prediction in zip(expected, predicted): matrix[int(actual), int(prediction)] += 1
    f1 = []
    for index in range(classes):
        tp = int(matrix[index,index]); fp = int(matrix[:,index].sum())-tp; fn = int(matrix[index].sum())-tp
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return {'accuracy':float(np.trace(matrix)/max(1,matrix.sum())), 'macro_f1':sum(f1)/classes,
            'per_class_f1':f1, 'confusion':matrix.tolist(), 'documents':int(matrix.sum())}
