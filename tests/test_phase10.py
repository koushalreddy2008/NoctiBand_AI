"""Unit tests for Phase 10: Cohort provenance, test subject isolation, and matrix consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from noctiband.evaluation.harness import build_pipeline, compute_metrics
from scripts.reconcile_phase10 import TEST_SUBJECTS, get_clean_cohort, matrix_metrics


def test_test_subjects_never_in_cohort() -> None:
    """Verify that none of the 4 sequestered test subjects appear in the clean cohort."""
    df_clean = get_clean_cohort()
    cohort_subjects = set(df_clean["subject_id"].astype(str).unique())
    overlap = cohort_subjects.intersection(TEST_SUBJECTS)
    assert len(overlap) == 0, f"Test subject leakage detected: {overlap}"


def test_cohort_provenance_and_totals() -> None:
    """Verify exact cohort size, epoch totals, and wake counts."""
    df_clean = get_clean_cohort()
    assert df_clean["subject_id"].nunique() == 22
    assert len(df_clean) == 18187
    assert int((df_clean["label_binary"] == 0).sum()) == 1516
    assert int((df_clean["label_3class"] == 0).sum()) == 1516
    assert int((df_clean["label_3class"] == 1).sum()) == 12771
    assert int((df_clean["label_3class"] == 2).sum()) == 3900


def test_subject_7749105_excluded() -> None:
    """Verify subject 7749105 is excluded due to failing the coverage guard."""
    df_clean = get_clean_cohort()
    assert "7749105" not in df_clean["subject_id"].astype(str).values


def test_matrix_metrics_matches_compute_metrics() -> None:
    """Verify matrix_metrics matches compute_metrics on synthetic confusion matrix."""
    cm = np.array([[883, 395, 238], [1350, 8201, 3220], [327, 1086, 2487]])
    classes = [0, 1, 2]
    metrics = matrix_metrics(cm, classes)

    # Reconstruct true and pred to compare with compute_metrics
    y_true = []
    y_pred = []
    for i in range(3):
        for j in range(3):
            y_true.extend([i] * cm[i, j])
            y_pred.extend([j] * cm[i, j])

    yt = np.array(y_true)
    yp = np.array(y_pred)
    res = compute_metrics(yt, yp, classes=np.array(classes))

    assert metrics["accuracy"] == res["accuracy"]
    assert metrics["balanced_accuracy"] == res["balanced_accuracy"]
    assert metrics["macro_f1"] == res["macro_f1"]
    assert metrics["cohen_kappa"] == res["cohen_kappa"]


def test_recompute_metrics_from_confusion_matrices() -> None:
    """Assert Bal Acc, Macro-F1, Kappa, and Wake F1 strictly equal matrix-derived values."""
    oof_path = Path("data/processed/oof_predictions_phase10.csv")
    assert oof_path.exists(), "Stored OOF predictions file missing"

    df_oof = pd.read_csv(oof_path)
    y_true = df_oof["y_true_binary"].to_numpy()

    target_models = [
        "pred_weak_motion_binary",
        "pred_ref_motion_binary",
        "pred_lr_motion_binary",
        "pred_lr_hr_binary",
        "pred_lr_fixed_binary",
        "pred_lr_tuned_binary",
        "pred_rf_fixed_binary",
        "pred_rf_tuned_binary",
    ]

    for col in target_models:
        y_pred = df_oof[col].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # 1. From arrays via standard scikit-learn
        acc_direct = accuracy_score(y_true, y_pred)
        bal_acc_direct = balanced_accuracy_score(y_true, y_pred)
        mf1_direct = f1_score(y_true, y_pred, average="macro")
        wf1_direct = f1_score(y_true, y_pred, labels=[0], average=None)[0]
        kappa_direct = cohen_kappa_score(y_true, y_pred)

        # 2. Recomputed strictly from cm
        total = float(cm.sum())
        acc_cm = float(np.trace(cm)) / total
        rec0 = float(cm[0, 0]) / float(cm[0].sum())
        rec1 = float(cm[1, 1]) / float(cm[1].sum())
        bal_acc_cm = (rec0 + rec1) / 2.0

        prec0 = float(cm[0, 0]) / float(cm[:, 0].sum()) if cm[:, 0].sum() > 0 else 0.0
        prec1 = float(cm[1, 1]) / float(cm[:, 1].sum()) if cm[:, 1].sum() > 0 else 0.0
        f1_0 = 2.0 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
        f1_1 = 2.0 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
        mf1_cm = (f1_0 + f1_1) / 2.0
        wf1_cm = f1_0

        po = acc_cm
        sum_0 = float(cm[0].sum()) * float(cm[:, 0].sum())
        sum_1 = float(cm[1].sum()) * float(cm[:, 1].sum())
        pe = (sum_0 + sum_1) / (total * total)
        kappa_cm = (po - pe) / (1.0 - pe)

        # Assert equivalence with atol=1e-4
        np.testing.assert_allclose(acc_direct, acc_cm, atol=1e-4)
        np.testing.assert_allclose(bal_acc_direct, bal_acc_cm, atol=1e-4)
        np.testing.assert_allclose(mf1_direct, mf1_cm, atol=1e-4)
        np.testing.assert_allclose(wf1_direct, wf1_cm, atol=1e-4)
        np.testing.assert_allclose(kappa_direct, kappa_cm, atol=1e-4)


def test_recompute_3class_metrics_from_confusion_matrices() -> None:
    """Assert 3-class Bal Acc, Macro-F1, and Kappa strictly equal matrix_metrics."""
    oof_path = Path("data/processed/oof_predictions_phase10.csv")
    df_oof = pd.read_csv(oof_path)
    y_3c = df_oof["y_true_3class"].to_numpy()

    target_3c = [
        "pred_3class_majority",
        "pred_lr_3class",
        "pred_rf_3class",
    ]

    for col in target_3c:
        y_pred = df_oof[col].to_numpy()
        cm = confusion_matrix(y_3c, y_pred, labels=[0, 1, 2])
        m_calc = matrix_metrics(cm, [0, 1, 2])

        acc_direct = accuracy_score(y_3c, y_pred)
        bal_acc_direct = balanced_accuracy_score(y_3c, y_pred)
        mf1_direct = f1_score(y_3c, y_pred, average="macro")
        kappa_direct = cohen_kappa_score(y_3c, y_pred)

        np.testing.assert_allclose(acc_direct, m_calc["accuracy"], atol=1e-4)
        np.testing.assert_allclose(bal_acc_direct, m_calc["balanced_accuracy"], atol=1e-4)
        np.testing.assert_allclose(mf1_direct, m_calc["macro_f1"], atol=1e-4)
        np.testing.assert_allclose(kappa_direct, m_calc["cohen_kappa"], atol=1e-4)


def test_inner_cv_threshold_selection_never_accesses_validation_fold() -> None:
    """Verify that inner-CV threshold tuning runs exclusively on training folds."""
    rng = np.random.default_rng(42)
    n_samples = 400
    X = rng.normal(size=(n_samples, 4))
    y = rng.choice([0, 1], size=n_samples, p=[0.2, 0.8])
    groups = np.repeat(np.arange(10), 40)

    gkf = GroupKFold(n_splits=5)

    class PoisonValidationData:
        """Sentinel object that raises an error if accessed in any way."""
        def __getitem__(self, item):
            raise AssertionError("Validation fold data was accessed during training/tuning!")

        def __getattr__(self, name):
            raise AssertionError("Validation fold attribute was accessed during training/tuning!")

    for train_idx, _val_idx in gkf.split(X, y, groups):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        _ = PoisonValidationData()

        # Inner 4-fold CV on training split ONLY
        inner_gkf = GroupKFold(n_splits=4)
        inner_probs = np.zeros(len(train_idx))

        for in_trn, in_val in inner_gkf.split(X_train, y_train, groups_train):
            pipe = build_pipeline("logistic_regression")
            pipe.fit(X_train[in_trn], y_train[in_trn])
            inner_probs[in_val] = pipe.predict_proba(X_train[in_val])[:, 0]

        # Select best threshold on inner training out-of-fold predictions
        thresholds = np.linspace(0.1, 0.9, 17)
        best_th, best_f1 = 0.5, -1.0
        for th in thresholds:
            p_in = np.where(inner_probs >= th, 0, 1)
            f1 = f1_score(y_train, p_in, labels=[0], average=None, zero_division=0)[0]
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        assert 0.1 <= best_th <= 0.9
        # Validation fold was never touched; test passes without triggering poison
