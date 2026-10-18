"""Tests for noctiband.preprocessing.epoching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noctiband.preprocessing.epoching import assign_epochs, epoch_coverage


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
