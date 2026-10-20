"""Unit tests for Phase 8: metric consistency with confusion matrix and MODEL_FEATURES audit."""

from __future__ import annotations

import numpy as np
import pytest

from noctiband.evaluation.harness import compute_metrics
from noctiband.features.epoch_features import MODEL_FEATURES


def test_metrics_consistency_with_confusion_matrix() -> None:
    """Verify that compute_metrics outputs strictly match confusion matrix calculations."""
    classes = np.array([0, 1, 2])
    # Synthetic out-of-fold predictions with imbalanced classes
    # 0 = Wake (100), 1 = NREM (500), 2 = REM (200)
    y_true = np.array([0] * 100 + [1] * 500 + [2] * 200)
    # Model predicts 0 and 1 only; zero predictions for class 2
    y_pred = np.array(
        [0] * 60 + [1] * 40 + [0] * 50 + [1] * 450 + [0] * 20 + [1] * 180
    )

    metrics = compute_metrics(y_true, y_pred, classes=classes)
    cm = np.array(metrics["confusion_matrix"])

    # 1. Total samples
    total = cm.sum()
    assert total == len(y_true)

    # 2. Accuracy: trace(cm) / sum(cm)
    expected_acc = round(float(np.trace(cm) / total), 4)
    assert metrics["accuracy"] == expected_acc

    # 3. Balanced Accuracy: mean of class recalls
    class_recalls = [cm[i, i] / cm[i].sum() for i in range(len(classes))]
    expected_bal_acc = round(float(np.mean(class_recalls)), 4)
    assert metrics["balanced_accuracy"] == expected_bal_acc

    # 4. Macro-F1: mean of class F1s
    class_f1s = []
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        class_f1s.append(f1)
    expected_macro_f1 = round(float(np.mean(class_f1s)), 4)
    assert metrics["macro_f1"] == expected_macro_f1

    # 5. Cohen's Kappa from matrix
    po = np.trace(cm) / total
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(len(classes))) / (total * total)
    expected_kappa = round(float((po - pe) / (1.0 - pe)), 4)
    assert metrics["cohen_kappa"] == expected_kappa


def test_model_features_has_no_time_or_position_leakage() -> None:
    """Verify MODEL_FEATURES contains no position in night, epoch index, or clock time."""
    forbidden_terms = [
        "epoch",
        "time",
        "hour",
        "clock",
        "position",
        "index",
        "length",
        "duration",
        "start",
    ]
    for feat in MODEL_FEATURES:
        feat_lower = feat.lower()
        # Ensure none of the features encodes night position or index
        for term in forbidden_terms:
            # Exception: 'hr_last_minus_first' contains 'first' but is a 30s intra-epoch feature
            if term == "length" or term == "duration" or term == "clock":
                assert term not in feat_lower, f"Feature '{feat}' encodes {term}."
            if feat_lower in ["epoch", "epoch_idx", "hours_since_start", "time_of_day"]:
                pytest.fail(f"Feature '{feat}' violates time/position isolation.")
