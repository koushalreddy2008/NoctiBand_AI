"""Unit tests for HRV feature extraction and spectral analysis (v0.2)."""

import numpy as np
import pandas as pd

from noctiband.features.hrv import (
    HRV_COLUMNS,
    compute_epoch_hrv_features,
    compute_hrv_metrics,
    enrich_with_hrv_features,
)


def test_compute_hrv_metrics_known_sequence():
    # Synthetic RR intervals in milliseconds: 1000ms (60 bpm) alternating with 1050ms
    rr = np.array([1000.0, 1050.0, 1000.0, 1050.0, 1000.0, 1050.0, 1000.0])
    metrics = compute_hrv_metrics(rr)

    for col in HRV_COLUMNS:
        assert col in metrics
        assert not np.isnan(metrics[col])

    # RMSSD of differences of exactly 50ms should equal 50.0
    assert metrics["hrv_rmssd"] == 50.0
    assert 0.0 <= metrics["hrv_pnn50"] <= 100.0
    assert metrics["hrv_sdnn"] > 0.0


def test_compute_hrv_metrics_too_few_samples():
    metrics = compute_hrv_metrics(np.array([1000.0, 1050.0]))
    assert np.isnan(metrics["hrv_rmssd"])
    assert np.isnan(metrics["hrv_sdnn"])


def test_compute_epoch_hrv_features_and_enrich():
    # 2 epochs (60 seconds) with 1 Hz heart rate
    rec_start = 1000.0
    t = np.arange(0, 60, 1.0)
    hr = 60.0 + 5.0 * np.sin(t)
    hr_df = pd.DataFrame({"timestamp": rec_start + t, "hr": hr})

    hrv_df = compute_epoch_hrv_features(hr_df, rec_start=rec_start, n_epochs=2, epoch_s=30)
    assert len(hrv_df) == 2
    for col in HRV_COLUMNS:
        assert col in hrv_df.columns
        assert not np.isnan(hrv_df.iloc[0][col])

    # Test enrich
    dummy_epoch_df = pd.DataFrame({"epoch": [0, 1], "stage": ["Wake", "Sleep"]})
    enriched = enrich_with_hrv_features(dummy_epoch_df, hr_df, rec_start=rec_start, epoch_s=30)
    assert "hrv_rmssd" in enriched.columns
    assert len(enriched) == 2
