"""Binary metrics. Positive class = unsafe (1), matching the paper's convention."""
from __future__ import annotations


def prf(golds: list[int], preds: list[int]) -> dict:
    assert len(golds) == len(preds)
    tp = sum(1 for g, p in zip(golds, preds) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(golds, preds) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(golds, preds) if g == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(golds),
        "n_pos_gold": sum(golds),
        "n_pred_unsafe": sum(preds),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
