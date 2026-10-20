"""Tests for Phase 7 fixes: deduplication, sampling-rate invariance, and MODEL_FEATURES."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from noctiband.features.epoch_features import (
    MODEL_FEATURES,
    QC_FEATURE_COLUMNS,
    build_epoch_table,
)
from noctiband.io.adapter_v2 import load_subject_v2
from noctiband.types import NightData


def test_model_features_excludes_qc_and_scaling_columns() -> None:
    """Verify MODEL_FEATURES strictly excludes all QC and sample-count columns."""
    for qc_col in QC_FEATURE_COLUMNS:
        assert (
            qc_col not in MODEL_FEATURES
        ), f"QC column '{qc_col}' must not be present in MODEL_FEATURES."

    # Verify explicitly mandated exclusions from prompt
    mandated_exclusions = [
        "hr_count",
        "motion_count",
        "hr_coverage",
        "motion_coverage",
        "hr_missing",
        "motion_missing",
        "hr_partial",
        "motion_partial",
        "partial_motion",
        "motion_activity_count",
    ]
    for col in mandated_exclusions:
        assert col not in MODEL_FEATURES, f"Mandated QC column '{col}' found in MODEL_FEATURES."


def test_sampling_rate_invariance_synthetic() -> None:
    """Test that converted motion features agree across 10, 33, and 50 Hz."""
    duration_s = 300.0  # 10 epochs

    def generate_signal(fs: float) -> pd.DataFrame:
        n_samples = int(round(fs * duration_s))
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        x = np.zeros(n_samples)
        y = np.zeros(n_samples)
        z = np.full(n_samples, -1.0)

        # Active window: 60s to 120s (epochs 2 and 3)
        active = (t >= 60.0) & (t < 120.0)
        x[active] = 0.2 * np.sin(2 * np.pi * 1.5 * t[active])
        y[active] = 0.1 * np.cos(2 * np.pi * 1.5 * t[active])
        z[active] = -1.0 + 0.15 * np.sin(2 * np.pi * 1.5 * t[active])
        return pd.DataFrame({"timestamp": t, "x": x, "y": y, "z": z})

    results = {}
    for fs in [10.0, 33.333, 50.0]:
        mot = generate_signal(fs)
        hr = pd.DataFrame({"timestamp": np.arange(0, 300, 5), "hr": np.full(60, 70.0)})
        night = NightData(
            subject="sim",
            night="1",
            hr=hr,
            motion=mot,
            labels=np.zeros(10, dtype=int),
            rec_start=0.0,
        )
        tbl = build_epoch_table(night)
        results[fs] = tbl

    # Features must agree within reasonable numerical tolerances
    for col in ["motion_mag_mean", "motion_mag_std", "motion_enmo_mean"]:
        diff_10 = np.abs(results[10.0][col] - results[50.0][col]).max()
        diff_33 = np.abs(results[33.333][col] - results[50.0][col]).max()
        assert diff_10 < 0.01, f"{col} 10 Hz diff {diff_10} exceeds tolerance"
        assert diff_33 < 0.01, f"{col} 33 Hz diff {diff_33} exceeds tolerance"

    # Per-sample fractions: active and still fraction agreement
    for col in ["motion_activity_fraction", "motion_fraction_still"]:
        diff_10 = np.abs(results[10.0][col] - results[50.0][col]).max()
        diff_33 = np.abs(results[33.333][col] - results[50.0][col]).max()
        assert diff_10 < 0.05, f"{col} 10 Hz fraction diff {diff_10} exceeds tolerance"
        assert diff_33 < 0.05, f"{col} 33 Hz fraction diff {diff_33} exceeds tolerance"


def test_deduplication_and_sorting_in_adapter(tmp_path: Path) -> None:
    """Test that adapter properly deduplicates duplicate HR timestamps and sorts."""
    base = tmp_path / "raw_v2_dedup"
    (base / "heart_rate").mkdir(parents=True)
    (base / "labels").mkdir(parents=True)
    (base / "motion").mkdir(parents=True)

    # 3 epochs
    (base / "labels" / "99_labeled_sleep.txt").write_text("0 0\n30 1\n60 2\n", encoding="utf-8")

    # HR with duplicate timestamps and non-monotonic order
    hr_text = "30.0,75\n10.0,70\n10.0,72\n5.0,68\n50.0,80\n50.0,80\n"
    (base / "heart_rate" / "99_heartrate.txt").write_text(hr_text, encoding="utf-8")

    # Motion with non-monotonic timestamps
    mot_text = (
        "10.0 0 0 -1\n"
        "5.0 0 0 -1\n"
        "15.0 0 0 -1\n"
        "25.0 0 0 -1\n"
        "20.0 0 0 -1\n"
        "40.0 0 0 -1\n"
        "70.0 0 0 -1\n"
    )
    (base / "motion" / "99_acceleration.txt").write_text(mot_text, encoding="utf-8")

    night = load_subject_v2(base, "99", allow_partial=True)

    # HR must be strictly monotonic without duplicate timestamps
    hr_ts = night.hr["timestamp"].to_numpy()
    assert (np.diff(hr_ts) > 0).all(), "HR timestamps must be strictly ascending and deduped."
    assert len(night.hr) == 4  # 5.0, 10.0, 30.0, 50.0

    # Motion must be strictly monotonic
    mot_ts = night.motion["timestamp"].to_numpy()
    assert (np.diff(mot_ts) > 0).all(), "Motion timestamps must be strictly ascending."


def test_recalibrated_epochs_since_last_high_activity() -> None:
    """Test recalibrated epochs_since_last_high_activity with cap and has_prior_activity."""
    # 5 epochs (150s): epoch 0,1 still; epoch 2 active; epoch 3,4 still
    n_samples = 1500  # 10 Hz for 150s
    t = np.linspace(0, 150, n_samples, endpoint=False)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    z = np.full(n_samples, -1.0)

    # Epoch 2 (60s to 90s): 15s of active motion (>0.05 ENMO) -> 50% active fraction (>=0.10)
    active = (t >= 65.0) & (t < 80.0)
    x[active] = 0.4
    y[active] = 0.4

    mot = pd.DataFrame({"timestamp": t, "x": x, "y": y, "z": z})
    hr = pd.DataFrame({"timestamp": np.arange(0, 150, 5), "hr": np.full(30, 70.0)})
    night = NightData(
        subject="recal",
        night="1",
        hr=hr,
        motion=mot,
        labels=np.zeros(5, dtype=int),
        rec_start=0.0,
    )
    tbl = build_epoch_table(night)

    # Epoch 0 and 1: before any high activity -> NaN and has_prior_activity == False
    assert np.isnan(tbl["epochs_since_last_high_activity"].iloc[0])
    assert not tbl["has_prior_activity"].iloc[0]
    assert np.isnan(tbl["epochs_since_last_high_activity"].iloc[1])
    assert not tbl["has_prior_activity"].iloc[1]

    # Epoch 2: high activity epoch -> 0.0 and has_prior_activity == True
    assert tbl["epochs_since_last_high_activity"].iloc[2] == 0.0
    assert tbl["has_prior_activity"].iloc[2]

    # Epoch 3: 1 epoch since high activity -> 1.0 and has_prior_activity == True
    assert tbl["epochs_since_last_high_activity"].iloc[3] == 1.0
    assert tbl["has_prior_activity"].iloc[3]

    # Epoch 4: 2 epochs since high activity -> 2.0 and has_prior_activity == True
    assert tbl["epochs_since_last_high_activity"].iloc[4] == 2.0
    assert tbl["has_prior_activity"].iloc[4]
