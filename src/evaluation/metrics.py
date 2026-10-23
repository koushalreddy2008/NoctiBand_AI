"""
Evaluation metrics for NoctiBand AI sleep-state classification models.

Computes comprehensive performance metrics adhering to clinical sleep validation
standards (not relying on raw accuracy alone): Macro F1, Balanced Accuracy,
Cohen's Kappa, per-class Precision/Recall/F1/Support, and Confusion Matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

DEFAULT_LABELS = (0, 1)
DEFAULT_NAMES = ("Wake", "Sleep")


def evaluate(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int] = DEFAULT_LABELS,
    names: Sequence[str] = DEFAULT_NAMES,
) -> dict:
    """Return every metric required by the project spec.

    Calculates Macro F1, Balanced Accuracy, Cohen's Kappa, Accuracy,
    per-class metrics (precision, recall, f1, support), and confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
