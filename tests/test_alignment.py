"""
Tests for timestamp mapping, alignment rule, boundary auditing, and windowing.
"""

import numpy as np
import pandas as pd

from src.preprocessing.alignment import (
    audit_signal_alignment,
    epoch_to_datetimes,
    epoch_to_timestamps,
    filter_and_align_signal,
    timestamp_to_epoch,
)


def test_timestamp_to_epoch_scalar():
    rec_start = 1000.0
    # Exact start
    assert timestamp_to_epoch(1000.0, rec_start) == 0
    # Mid-epoch
    assert timestamp_to_epoch(1015.0, rec_start) == 0
    # End boundary (half-open, belongs to epoch 0)
    assert timestamp_to_epoch(1029.999, rec_start) == 0
    # Exactly epoch 1 start
    assert timestamp_to_epoch(1030.0, rec_start) == 1
    # Before recStart
    assert timestamp_to_epoch(999.0, rec_start) == -1
    assert timestamp_to_epoch(970.0, rec_start) == -1
    assert timestamp_to_epoch(969.0, rec_start) == -2


def test_timestamp_to_epoch_array_and_series():
    rec_start = 1000.0
    ts_array = np.array([970.0, 1000.0, 1029.0, 1030.0, 1060.0])
    epochs = timestamp_to_epoch(ts_array, rec_start)
    np.testing.assert_array_equal(epochs, np.array([-1, 0, 0, 1, 2]))

    ts_series = pd.Series([1000.0, 1035.0])
    epochs_series = timestamp_to_epoch(ts_series, rec_start)
    assert list(epochs_series) == [0, 1]


def test_epoch_to_timestamps():
    rec_start = 1000.0
    start, end = epoch_to_timestamps(0, rec_start, epoch_duration_sec=30.0)
    assert start == 1000.0
    assert end == 1030.0

    start_5, end_5 = epoch_to_timestamps(5, rec_start, epoch_duration_sec=30.0)
    assert start_5 == 1150.0
    assert end_5 == 1180.0


def test_epoch_to_datetimes():
    rec_start = pd.Timestamp("2021-12-04 23:40:29", tz="America/New_York")
    dt_start, dt_end = epoch_to_datetimes(1, rec_start, epoch_duration_sec=30.0)
    assert dt_start == rec_start + pd.Timedelta(seconds=30)
    assert dt_end == rec_start + pd.Timedelta(seconds=60)
    assert dt_start.tzinfo == rec_start.tzinfo


def test_audit_signal_alignment():
    rec_start = 1000.0
    # 5 total epochs: [0, 1, 2, 3, 4]
    # Epoch 0: 1000-1030, Epoch 1: 1030-1060, Epoch 2: 1060-1090
    # Epoch 3: 1090-1120, Epoch 4: 1120-1150
    df = pd.DataFrame(
        {
            "timestamp": [
                980.0,   # epoch -1 (before start)
                995.0,   # epoch -1 (before start)
                1005.0,  # epoch 0 (inside)
                1010.0,  # epoch 0 (inside)
                1040.0,  # epoch 1 (inside)
                1100.0,  # epoch 3 (inside)
                1160.0,  # epoch 5 (after window)
            ]
        }
    )

    audit = audit_signal_alignment(df, rec_start, total_epochs=5)
    assert audit["total_samples"] == 7
    assert audit["samples_before_start"] == 2
    assert audit["samples_inside_window"] == 4
    assert audit["samples_after_window"] == 1
    # Inside epochs present: 0, 1, 3 -> missing: 2, 4
    assert audit["epochs_covered"] == 3
    assert audit["epochs_missing"] == 2
    assert audit["missing_epoch_indices"] == [2, 4]
    assert audit["coverage_ratio"] == 3 / 5


def test_filter_and_align_signal():
    rec_start = 1000.0
    df = pd.DataFrame({"timestamp": [990.0, 1005.0, 1045.0, 1160.0], "val": [1, 2, 3, 4]})

    # Drop outside [0, 2]
    filtered = filter_and_align_signal(df, rec_start, total_epochs=3, drop_outside=True)
    assert len(filtered) == 2
    assert list(filtered["epoch"]) == [0, 1]

    # Keep all
    all_rows = filter_and_align_signal(df, rec_start, total_epochs=3, drop_outside=False)
    assert len(all_rows) == 4
    assert list(all_rows["epoch"]) == [-1, 0, 1, 5]
