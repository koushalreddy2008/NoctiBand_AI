"""
Temporal smoothing and post-processing filters for sleep stage predictions.
Written by Person A (Senior ML / Pipeline Engineer).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def median_smooth(p_sleep: np.ndarray | list[float], window: int) -> np.ndarray:
    """Apply 1D median filter to sleep probabilities or predictions for a single night.

    Parameters
    ----------
    p_sleep : array-like
        Array or sequence of sleep probabilities (or binary stages) for consecutive epochs.
    window : int
        Odd integer window length. A window of 1 represents no smoothing.

    Returns
    -------
    np.ndarray
        Smoothed float array of identical shape.
    """
    p = np.asarray(p_sleep, dtype=float)
    if len(p) == 0:
        return p.copy()

    if window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")

    if window == 1:
        return p.copy()

    if window % 2 == 0:
        raise ValueError(f"window must be an odd integer, got {window}")

    return median_filter(p, size=window, mode="nearest")


def smooth_probabilities(
    probs: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Multi-class or 1D array probability smoother."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        return median_smooth(arr, window_size)
    elif arr.ndim == 2:
        smoothed = np.zeros_like(arr)
        for c in range(arr.shape[1]):
            smoothed[:, c] = median_smooth(arr[:, c], window_size)
        sums = smoothed.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return smoothed / sums
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")
