"""Unit tests for temporal smoothing module (Stage 2)."""

import numpy as np
import pytest

from noctiband.models.smoothing import (
    TemporalSmoother,
    smooth_labels,
    smooth_probabilities,
)


def test_smooth_labels_identity_k1():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    smoothed = smooth_labels(y, window_size=1)
    np.testing.assert_array_equal(smoothed, y)


def test_smooth_labels_removes_single_glitch():
    # Single isolated 0 in a sea of 1s
    y = np.array([1, 1, 1, 0, 1, 1, 1])
    smoothed = smooth_labels(y, window_size=3)
    expected = np.array([1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(smoothed, expected)

    # Single isolated 1 in a sea of 0s
    y2 = np.array([0, 0, 0, 1, 0, 0, 0])
    smoothed2 = smooth_labels(y2, window_size=3)
    np.testing.assert_array_equal(smoothed2, np.zeros(7, dtype=int))


def test_smooth_probabilities_shape_and_bounds():
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.0, 1.0, size=(100, 2))
    smoothed = smooth_probabilities(probs, window_size=5)
    assert smoothed.shape == (100, 2)
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 1.0)
    # Probabilities should still sum close to 1 across classes
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(100), atol=1e-5)


def test_invalid_window_size():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="must be a positive odd integer"):
        smooth_labels(y, window_size=2)
    with pytest.raises(ValueError, match="must be a positive odd integer"):
        smooth_labels(y, window_size=0)


def test_temporal_smoother_sklearn_interface():
    smoother = TemporalSmoother(window_size=3)
    y = np.array([1, 1, 0, 1, 1])
    res = smoother.fit_transform(y)
    np.testing.assert_array_equal(res, np.array([1, 1, 1, 1, 1]))
    assert smoother.get_params()["window_size"] == 3
