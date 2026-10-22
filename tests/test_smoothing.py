"""
Unit tests for postprocess smoothing module and per-night smoothing.
"""

import numpy as np
import pandas as pd
import pytest

from noctiband.evaluation.tuning import smooth_per_night
from noctiband.postprocess.smoothing import median_smooth, smooth_probabilities


def test_median_smooth_window_1_identity():
    arr = np.array([0.1, 0.9, 0.2, 0.8, 0.4])
    res = median_smooth(arr, window=1)
    np.testing.assert_array_equal(res, arr)


def test_median_smooth_removes_single_wake_blip():
    # Model predicts [Sleep, Sleep, Wake, Sleep, Sleep]
    p_sleep = np.array([0.9, 0.85, 0.1, 0.92, 0.88])
    res = median_smooth(p_sleep, window=3)
    assert res[2] > 0.5  # Lone blip smoothed above sleep threshold
    np.testing.assert_allclose(res[2], 0.85)


def test_median_smooth_invalid_window():
    arr = np.array([0.5, 0.6])
    with pytest.raises(ValueError, match="window must be an odd integer"):
        median_smooth(arr, window=2)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        median_smooth(arr, window=0)


def test_median_smooth_empty_array():
    arr = np.array([])
    res = median_smooth(arr, window=3)
    assert len(res) == 0


def test_smooth_probabilities_2d():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.8, 0.2]])
    smoothed = smooth_probabilities(probs, window_size=3)
    assert smoothed.shape == (3, 2)
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(3), atol=1e-5)


def test_smooth_per_night():
    df = pd.DataFrame({
        "subject_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "night_id": ["n1", "n1", "n1", "n1", "n1", "n1"],
        "epoch_idx": [0, 1, 2, 0, 1, 2],
    })
    p_sleep = np.array([0.9, 0.1, 0.9, 0.2, 0.8, 0.2])
    smoothed = smooth_per_night(df, p_sleep, window=3)
    assert len(smoothed) == 6
    # For subject s1, the middle 0.1 should be smoothed to 0.9
    assert smoothed[1] == 0.9
    # For subject s2, the middle 0.8 should be smoothed to 0.2
    assert smoothed[4] == 0.2
