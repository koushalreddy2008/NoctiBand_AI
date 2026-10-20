"""
Tests for model evaluation metrics.
"""

import pytest

from noctiband.evaluation.metrics import evaluate


def test_evaluate_known_example():
    res = evaluate([0, 0, 1, 1], [0, 1, 1, 1])

    assert res["confusion_matrix"] == [[1, 1], [0, 2]]
    assert res["balanced_accuracy"] == pytest.approx(0.75)
    assert res["kappa"] == pytest.approx(0.5)
    assert res["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert res["per_class"]["Wake"]["recall"] == pytest.approx(0.5)


def test_evaluate_perfect_prediction():
    res = evaluate([0, 1, 0, 1], [0, 1, 0, 1])

    assert res["accuracy"] == 1.0
    assert res["balanced_accuracy"] == 1.0
    assert res["kappa"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["per_class"]["Wake"]["precision"] == 1.0
    assert res["per_class"]["Sleep"]["recall"] == 1.0


def test_evaluate_multiclass_3stage():
    labels = [0, 1, 2]
    names = ["Wake", "NREM", "REM"]

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]

    res = evaluate(y_true, y_pred, labels=labels, names=names)

    assert len(res["confusion_matrix"]) == 3
    assert "Wake" in res["per_class"]
    assert "NREM" in res["per_class"]
    assert "REM" in res["per_class"]
    assert res["per_class"]["Wake"]["precision"] == 1.0