"""Unit tests for chunked epoching and vectorized strided aggregations."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from noctiband.io.contracts import NightRecording
from noctiband.preprocessing.epoching import (
    align_night_to_epochs,
    compute_strided_epoch_aggregations,
    iter_align_night_chunks,
)


def test_compute_strided_epoch_aggregations():
    # 60 samples, 20 samples per epoch -> exactly 3 epochs
    data = np.arange(60, dtype=float)
    res = compute_strided_epoch_aggregations(data, samples_per_epoch=20)

    assert len(res["mean"]) == 3
    assert res["mean"][0] == np.mean(data[:20])
    assert res["mean"][1] == np.mean(data[20:40])
    assert res["mean"][2] == np.mean(data[40:60])
    assert res["min"][0] == 0.0
    assert res["max"][2] == 59.0
    assert list(res["count"]) == [20, 20, 20]


def test_compute_strided_empty_or_zero():
    res = compute_strided_epoch_aggregations(np.array([]), samples_per_epoch=20)
    assert len(res["mean"]) == 0

    res_zero = compute_strided_epoch_aggregations(np.arange(10), samples_per_epoch=0)
    assert len(res_zero["mean"]) == 0


def test_chunked_epoch_generator():
    rec_start = datetime(2026, 10, 22, 22, 0, 0, tzinfo=UTC)
    # 6 epochs total (180 seconds)
    total_sec = 180
    t_hr = np.arange(0, total_sec, 1.0)
    hr_df = pd.DataFrame({
        "timestamp": rec_start.timestamp() + t_hr,
        "heart_rate": 60.0 + 5.0 * np.sin(t_hr),
    })

    t_mot = np.arange(0, total_sec, 0.1)
    motion_df = pd.DataFrame({
        "timestamp": rec_start.timestamp() + t_mot,
        "x": 0.02 * np.ones(len(t_mot)),
        "y": 0.02 * np.ones(len(t_mot)),
        "z": 0.98 * np.ones(len(t_mot)),
    })

    recording = NightRecording(
        subject_id="test_sub",
        night_id="test_night",
        rec_start=pd.to_datetime(rec_start),
        epoch_duration_sec=30.0,
        hr=hr_df,
        motion=motion_df,
        labels=np.array([0, 1, 1, 2, 2, 0]),
        metadata={},
    )

    # Stream in chunks of 2 epochs
    chunks = list(iter_align_night_chunks(recording, chunk_size_epochs=2))
    assert len(chunks) == 3
    assert len(chunks[0].epochs_df) == 2
    assert list(chunks[0].epochs_df["epoch"]) == [0, 1]
    assert list(chunks[1].epochs_df["epoch"]) == [2, 3]
    assert list(chunks[2].epochs_df["epoch"]) == [4, 5]

    # Compare full alignment vs chunked alignment
    full_epoched = align_night_to_epochs(recording, chunk_size_epochs=None)
    chunked_epoched = align_night_to_epochs(recording, chunk_size_epochs=2)

    assert len(full_epoched.epochs_df) == 6
    assert len(chunked_epoched.epochs_df) == 6
    np.testing.assert_array_almost_equal(
        full_epoched.epochs_df["hr_mean"].dropna().values,
        chunked_epoched.epochs_df["hr_mean"].dropna().values,
    )
