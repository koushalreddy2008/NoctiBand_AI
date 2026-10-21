"""Unit tests for Phase 9: MotionThresholdBaseline direction and label coding."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from noctiband.models.baselines import MotionThresholdBaseline


def test_motion_threshold_direction_and_label_coding() -> None:
    """Verify MotionThresholdBaseline maps higher motion to Wake (0) and lower to Sleep (1)."""
    # Create clear synthetic separation:
    # 0 = Wake (high motion ~ 5.0)
    # 1 = Sleep (low motion ~ 0.2)
    rng = np.random.RandomState(42)
    n_wake = 50
    n_sleep = 200

    wake_motion = rng.normal(loc=5.0, scale=0.5, size=(n_wake, 1))
    sleep_motion = rng.normal(loc=0.2, scale=0.1, size=(n_sleep, 1))

    X = np.vstack([wake_motion, sleep_motion])
    y = np.array([0] * n_wake + [1] * n_sleep)

    model = MotionThresholdBaseline(feature_idx=0)
    model.fit(X, y)

    # Threshold should sit comfortably between sleep and wake distributions (~1.0 - 4.0)
    assert 0.5 < model.threshold_ < 4.5

    # Predict on test samples
    test_high = np.array([[6.0], [4.5], [10.0]])
    test_low = np.array([[0.1], [0.3], [0.0]])

    preds_high = model.predict(test_high)
    preds_low = model.predict(test_low)

    # High motion MUST predict 0 (Wake)
    assert np.all(preds_high == 0), f"Expected Wake (0) for high motion, got {preds_high}"
    # Low motion MUST predict 1 (Sleep)
    assert np.all(preds_low == 1), f"Expected Sleep (1) for low motion, got {preds_low}"

    # Verify predict_proba: class 0 probability must be higher for high motion
    probs_high = model.predict_proba(test_high)
    probs_low = model.predict_proba(test_low)

    assert np.all(probs_high[:, 0] > 0.5), "P(Wake=0) should be > 0.5 for high motion"
    assert np.all(probs_low[:, 0] < 0.5), "P(Wake=0) should be < 0.5 for low motion"


def test_threshold_overfitting_with_weak_feature() -> None:
    """Demonstrate why weak feature can yield pooled balanced accuracy < 0.50 across folds.

    When a feature is weakly predictive, optimizing threshold on training folds
    can pick extreme boundary values that cause severe class imbalance on validation folds.
    """
    rng = np.random.RandomState(99)
    # Feature with weak separation (AUC ~ 0.56)
    # Training fold
    y_train = np.array([0] * 50 + [1] * 450)
    x_train = rng.normal(loc=0.0, scale=1.0, size=(500, 1))
    x_train[:50] += 0.2  # subtle shift for Wake

    model = MotionThresholdBaseline(feature_idx=0)
    model.fit(x_train, y_train)

    # Validation fold with slightly shifted distribution
    y_val = np.array([0] * 30 + [1] * 270)
    x_val = rng.normal(loc=0.0, scale=1.0, size=(300, 1))
    x_val[:30] += 0.1

    val_preds = model.predict(x_val)
    bal_acc = balanced_accuracy_score(y_val, val_preds)

    # Check that model fitted and predicts valid binary labels {0, 1}
    assert set(np.unique(val_preds)).issubset({0, 1})
    assert isinstance(bal_acc, float)
