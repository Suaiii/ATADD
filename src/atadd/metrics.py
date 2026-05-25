from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

KNOWN_TYPES = ("speech", "sound", "singing", "music")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    eps = 1e-12
    f1_list = []
    for c in range(num_classes):
        tp = np.logical_and(y_true == c, y_pred == c).sum()
        fp = np.logical_and(y_true != c, y_pred == c).sum()
        fn = np.logical_and(y_true == c, y_pred != c).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_list.append(f1)
    return float(np.mean(f1_list))


def type_macro_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_types: Iterable[Optional[str]],
    num_classes: int,
) -> Dict[str, float]:
    sample_types = np.asarray(list(sample_types), dtype=object)
    metrics: Dict[str, float] = {}
    present_scores = []
    for audio_type in KNOWN_TYPES:
        mask = sample_types == audio_type
        if not np.any(mask):
            continue
        score = macro_f1_score(y_true[mask], y_pred[mask], num_classes=num_classes)
        metrics[f"{audio_type}_macro_f1"] = score
        present_scores.append(score)
    if present_scores:
        metrics["track2_macro_f1"] = float(np.mean(present_scores))
    return metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    sample_types: Optional[Iterable[Optional[str]]] = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred, num_classes=num_classes),
    }
    if sample_types is not None:
        metrics.update(type_macro_f1_scores(y_true, y_pred, sample_types=sample_types, num_classes=num_classes))
    return metrics
