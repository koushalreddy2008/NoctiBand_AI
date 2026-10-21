"""Tests for Stage 3 end-to-end inference and report bridge."""

import numpy as np
import pandas as pd
import pytest

from noctiband.inference.predict import predict_night
from noctiband.report_bridge import (
    DISCLAIMER,
    compute_sleep_metrics,
    make_report,
)
from noctiband.types import NightData, NightPrediction, QualityControlError


@pytest.fixture
def synthetic_8h_night() -> NightData:
    """Create a synthetic 8-hour night with 50 Hz motion and 1 Hz HR."""
    n_epochs = 960  # 8 hours = 480 min = 960 epochs of 30s
    total_seconds = n_epochs * 30

    # 1 Hz HR: 1 reading every second
    t_hr = np.arange(0, total_seconds, 1.0)
    hr_vals = 60.0 + 5.0 * np.sin(2 * np.pi * t_hr / 3600.0)
    hr_df = pd.DataFrame({"timestamp": t_hr, "hr": hr_vals})

    # 50 Hz motion
    t_mot = np.arange(0, total_seconds, 0.02)
    x = 0.05 * np.sin(2 * np.pi * t_mot / 10.0)
    y = 0.05 * np.cos(2 * np.pi * t_mot / 10.0)
    z = 0.98 + 0.02 * np.sin(2 * np.pi * t_mot / 5.0)
    mot_df = pd.DataFrame({"timestamp": t_mot, "x": x, "y": y, "z": z})

    return NightData(
        subject="test_synthetic_sub",
        night="1",
        hr=hr_df,
        motion=mot_df,
        labels=None,
        rec_start=0.0,
    )


def test_predict_night_e2e_shape(synthetic_8h_night):
    pred = predict_night(synthetic_8h_night, allow_partial=True)
    assert isinstance(pred, NightPrediction)
    assert len(pred.epoch_labels) == 960
    assert len(pred.probs) == 960
    assert len(pred.coverage) == 960
    assert len(pred.epoch_timestamps) == 960
    assert pred.qc_summary is not None
    assert pred.rec_start == 0.0
    # Hypnogram property
    np.testing.assert_array_equal(pred.hypnogram, pred.epoch_labels)


def test_predict_night_qc_guard():
    # Artificially sparse samples spanning 2000s to test QC guard (< 95% coverage)
    t_sparse = np.array([0.0, 1000.0, 2000.0])
    hr_sparse = pd.DataFrame({"timestamp": t_sparse, "hr": [60.0, 65.0, 62.0]})
    mot_sparse = pd.DataFrame({
        "timestamp": t_sparse,
        "x": [0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0],
        "z": [1.0, 1.0, 1.0],
    })

    night_sparse = NightData(
        subject="sparse_sub",
        night="1",
        hr=hr_sparse,
        motion=mot_sparse,
        labels=None,
        rec_start=0.0,
    )

    with pytest.raises(QualityControlError, match="quality control guard"):
        predict_night(night_sparse, allow_partial=False)

    # When allow_partial=True, should not raise, but flag in qc_summary
    pred_allowed = predict_night(night_sparse, allow_partial=True)
    assert pred_allowed.qc_summary["passed_qc"] is False
    assert len(pred_allowed.qc_summary["warnings"]) > 0


def test_sleep_metrics_calculation():
    # 10 epochs: 2 wake, 5 sleep, 2 wake (WASO), 1 sleep
    # Sequence: [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    m = compute_sleep_metrics(labels)

    # Total epochs = 10, total window = 5.0 min
    assert m["total_window_min"] == 5.0
    # Sleep epochs = 6 -> TST = 3.0 min
    assert m["tst_min"] == 3.0
    # SOL: sleep onset is index 2 -> 2 * 0.5 = 1.0 min
    assert m["sol_min"] == 1.0
    # WASO: between onset (idx 2) and final sleep (idx 9), there are 2 wake epochs at idx 5, 6
    assert m["waso_min"] == 1.0
    # Sleep Efficiency: 3.0 / 5.0 * 100 = 60.0%
    assert m["sleep_efficiency"] == 60.0
    # Transitions from sleep (1) to wake (0) in sleep window: exactly 1 transition
    assert m["noa"] == 1


def test_make_report_and_safety_disclaimer():
    labels = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0])
    pred = NightPrediction(
        rec_start=1600000000.0,
        epoch_labels=labels,
        probs=np.ones(10),
        coverage=np.ones(10),
    )
    report = make_report(pred, subject_id="Sub01", night_id="Night 1")

    assert report["subject_id"] == "Sub01"
    assert report["night_id"] == "Night 1"
    assert "metrics" in report
    assert "recommendations" in report
    assert report["disclaimer"] == DISCLAIMER

    # Non-diagnostic safety assertion: verify forbidden diagnostic words are absent
    forbidden_words = ["diagnos", "insomnia", "apnea", "pathology", "disease", "prescrib"]
    for tip in report["recommendations"]:
        tip_lower = tip.lower()
        for fw in forbidden_words:
            assert fw not in tip_lower, f"Forbidden word '{fw}' found in hygiene tip: '{tip}'"
