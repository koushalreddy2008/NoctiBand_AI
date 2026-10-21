"""Unit tests for Phase 10: Cohort provenance, test subject isolation, and matrix consistency."""

from __future__ import annotations

import numpy as np

from noctiband.evaluation.harness import compute_metrics
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
