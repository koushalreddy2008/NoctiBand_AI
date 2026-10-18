"""Tests for 30-second epoching, grid construction, aggregation, and signal quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noctiband.preprocessing.epoching import assign_epochs, epoch_coverage
from src.io.contracts import NightRecording, SignalQuality, SleepStage
from src.preprocessing.epoching import EpochAligner, align_night_to_epochs


def test_align_night_preserves_missing_epochs():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    rec_unix = rec_start.timestamp()

    # Create HR with readings only in epoch 0:
    # epoch 0: [rec_unix, rec_unix + 30)
    # epoch 1: [rec_unix + 30, rec_unix + 60) -> NO HR readings
    # epoch 2: [rec_unix + 60, rec_unix + 90) -> NO HR readings
    hr_df = pd.DataFrame(
        {
            "timestamp": [rec_unix + 5.0, rec_unix + 15.0, rec_unix + 25.0],
            "heart_rate": [60.0, 62.0, 64.0],
        }
    )
    labels = np.array([0, 2, 4])  # Wake, N2, REM

    rec = NightRecording(
        subject_id="sub_test",
        night_id="night_01",
        rec_start=rec_start,
        hr=hr_df,
        labels=labels,
    )

    aligner = EpochAligner(epoch_duration_sec=30.0)
    epoched = aligner.align_night(rec, preserve_slices=True)

    df = epoched.epochs_df
    # Crucial assertion: all 3 epochs must be present
    assert len(df) == 3
    assert list(df["epoch"]) == [0, 1, 2]

    # Epoch 0 checks
    assert df.loc[0, "hr_count"] == 3
    assert df.loc[0, "hr_mean"] == 62.0
    assert df.loc[0, "hr_min"] == 60.0
    assert df.loc[0, "hr_max"] == 64.0
    assert df.loc[0, "hr_range"] == 4.0
    assert df.loc[0, "hr_change"] == 4.0  # 64 - 60
    assert df.loc[0, "has_hr"] is True or df.loc[0, "has_hr"] == 1
    assert df.loc[0, "quality_flag"] == SignalQuality.GOOD.value
    assert df.loc[0, "sleep_stage"] == SleepStage.WAKE
    assert df.loc[0, "sleep_stage_name"] == "Wake"
    assert df.loc[0, "sleep_binary"] == 0
    assert df.loc[0, "sleep_3class"] == 0

    # Epoch 1 & 2 checks (Missing HR)
    assert df.loc[1, "hr_count"] == 0
    assert pd.isna(df.loc[1, "hr_mean"])
    assert df.loc[1, "has_hr"] is False or df.loc[1, "has_hr"] == 0
    assert df.loc[1, "quality_flag"] == SignalQuality.EMPTY.value  # No sensors at all in epoch 1
    assert df.loc[1, "sleep_stage"] == SleepStage.N2
    assert df.loc[1, "sleep_stage_name"] == "N2 (Core NREM)"
    assert df.loc[1, "sleep_binary"] == 1
    assert df.loc[1, "sleep_3class"] == 1

    # Check preserved slices
    slice_0 = epoched.get_epoch_slice(0)
    assert len(slice_0["hr"]) == 3
    slice_1 = epoched.get_epoch_slice(1)
    assert len(slice_1["hr"]) == 0


def test_align_night_multimodal_motion_bp_spo2():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    rec_unix = rec_start.timestamp()

    # 1 epoch
    hr_df = pd.DataFrame(
        {"timestamp": [rec_unix + 10.0, rec_unix + 20.0], "heart_rate": [70.0, 72.0]}
    )
    # Acceleration in g: x=0, y=0.6, z=0.8 -> VM = sqrt(0.36 + 0.64) = 1.0 -> ENMO = 0.0
    motion_df = pd.DataFrame(
        {
            "timestamp": [rec_unix + 5.0, rec_unix + 15.0],
            "x": [0.0, 0.0],
            "y": [0.6, 0.6],
            "z": [0.8, 0.8],
        }
    )
    bp_df = pd.DataFrame(
        {
            "timestamp": [rec_unix + 10.0],
            "systolic": [120.0],
            "diastolic": [80.0],
            "map": [93.3],
            "pulse_pressure": [40.0],
        }
    )
    spo2_df = pd.DataFrame(
        {
            "timestamp": [rec_unix + 5.0, rec_unix + 15.0],
            "spo2": [97.0, 93.0],  # one reading < 95
        }
    )

    rec = NightRecording(
        subject_id="sub_multi",
        night_id="night_01",
        rec_start=rec_start,
        hr=hr_df,
        motion=motion_df,
        bp=bp_df,
        spo2=spo2_df,
        labels=np.array([4]),  # REM
    )

    epoched = align_night_to_epochs(rec)
    df = epoched.epochs_df

    assert len(df) == 1
    row = df.iloc[0]
    assert row["hr_count"] == 2
    assert row["motion_count"] == 2
    assert pytest.approx(row["vm_mean"], 1e-4) == 1.0
    assert pytest.approx(row["enmo_mean"], 1e-4) == 0.0
    assert row["sbp_mean"] == 120.0
    assert row["dbp_mean"] == 80.0
    assert row["spo2_count"] == 2
    assert row["spo2_sec_below_95"] == 1
    assert row["quality_flag"] == SignalQuality.GOOD.value


def test_degraded_quality_flag_on_sparse_hr():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    rec_unix = rec_start.timestamp()

    # Only 1 HR reading in 30s epoch -> DEGRADED
    hr_df = pd.DataFrame({"timestamp": [rec_unix + 10.0], "heart_rate": [60.0]})
    rec = NightRecording(
        subject_id="sub_sparse",
        night_id="night_01",
        rec_start=rec_start,
        hr=hr_df,
        labels=np.array([1]),
    )

    epoched = align_night_to_epochs(rec)
    assert epoched.epochs_df.iloc[0]["quality_flag"] == SignalQuality.DEGRADED.value


def test_assign_epochs_boundaries():
    rec_start = 1000.0
    n_epochs = 10  # Window: [1000.0, 1300.0)

    timestamps = [
        rec_start - 10.0,         # Before window -> exclude
        rec_start - 0.001,        # Just before rec_start -> exclude
        rec_start,                # Exactly rec_start -> epoch 0
        rec_start + 0.001,        # Inside epoch 0 -> epoch 0
        rec_start + 29.999,       # End of epoch 0 -> epoch 0
        rec_start + 30.0,         # Exactly rec_start + 30 -> epoch 1
        rec_start + 30.001,       # Inside epoch 1 -> epoch 1
        rec_start + 60.0,         # Exactly rec_start + 60 -> epoch 2
        rec_start + 30.0 * 9,     # Exactly epoch 9 start -> epoch 9
        rec_start + 30.0 * 10 - 1e-6,  # Upper boundary interior -> epoch 9
        rec_start + 30.0 * 10,    # Exactly upper boundary (1300.0) -> exclude
        rec_start + 30.0 * 10 + 5.0,   # After window -> exclude
    ]

    df = pd.DataFrame({"timestamp": timestamps, "val": range(len(timestamps))})
    assigned = assign_epochs(df, rec_start=rec_start, n_epochs=n_epochs)

    # Expected retained epochs
    expected_epochs = [0, 0, 0, 1, 1, 2, 9, 9]
    assert list(assigned["epoch"]) == expected_epochs
    assert assigned["epoch"].dtype == int
    assert len(assigned) == len(expected_epochs)


def test_assign_epochs_no_clipping():
    """Verify readings outside [0, n_epochs) are never clipped into it."""
    rec_start = 0.0
    n_epochs = 5  # Window: [0, 150)

    df = pd.DataFrame({
        "timestamp": [-100.0, -1.0, 0.0, 50.0, 149.9, 150.0, 200.0],
        "signal": [1, 2, 3, 4, 5, 6, 7],
    })
    assigned = assign_epochs(df, rec_start, n_epochs)

    # Only timestamps 0.0, 50.0, 149.9 are inside [0, 150)
    assert list(assigned["timestamp"]) == [0.0, 50.0, 149.9]
    assert list(assigned["epoch"]) == [0, 1, 4]
    assert set(assigned["epoch"]).issubset(set(range(n_epochs)))


def test_assign_epochs_empty_and_errors():
    rec_start = 1000.0
    n_epochs = 10

    # Missing timestamp column
    df_invalid = pd.DataFrame({"time": [1000.0, 1010.0]})
    with pytest.raises(ValueError, match="must contain a 'timestamp' column"):
        assign_epochs(df_invalid, rec_start, n_epochs)

    # Empty DataFrame
    df_empty = pd.DataFrame({"timestamp": []})
    res_empty = assign_epochs(df_empty, rec_start, n_epochs)
    assert res_empty.empty
    assert "epoch" in res_empty.columns


def test_epoch_coverage_counts():
    rec_start = 0.0
    n_epochs = 5

    # Epoch 0: 3 readings; Epoch 1: 1 reading; Epoch 2: 0 readings;
    # Epoch 3: 2 readings; Epoch 4: 0 readings
    df = pd.DataFrame({
        "timestamp": [
            0.0, 10.0, 20.0,  # Epoch 0
            40.0,             # Epoch 1
            95.0, 110.0,      # Epoch 3
        ]
    })
    assigned = assign_epochs(df, rec_start, n_epochs)
    coverage = epoch_coverage(assigned, n_epochs)

    assert isinstance(coverage, np.ndarray)
    assert coverage.shape == (n_epochs,)
    assert coverage.dtype == int
    assert list(coverage) == [3, 1, 0, 2, 0]
    assert coverage.sum() == len(assigned)


def test_epoch_coverage_missing_column():
    df = pd.DataFrame({"timestamp": [10.0, 20.0]})
    with pytest.raises(ValueError, match="must contain an 'epoch' column"):
        epoch_coverage(df, 5)


def test_epoch_coverage_empty():
    df = pd.DataFrame({"epoch": []})
    coverage = epoch_coverage(df, 4)
    assert list(coverage) == [0, 0, 0, 0]
