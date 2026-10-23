"""Temporal post-processing and smoothing for sleep stage predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

from noctiband.postprocess.smoothing import median_smooth

__all__ = [
    "TemporalSmoother",
    "median_smooth",
    "smooth_labels",
    "smooth_probabilities",
    "tune_smoother_on_oof",
]


def smooth_probabilities(
    probs: np.ndarray,
    window_size: int = 3,
    causal: bool = False,
) -> np.ndarray:
    """Apply temporal smoothing to 1D or 2D probability arrays.

    Parameters
    ----------
    probs : np.ndarray
        Array of predicted probabilities (shape (N,) or (N, C)).
    window_size : int, default=3
        Odd integer kernel size. If <= 1, returns original probabilities.
    causal : bool, default=False
        If True, only use current and past epochs (strictly online/causal).
        If False, uses centered offline window.

    Returns
    -------
    np.ndarray
        Smoothed probabilities of same shape.
    """
    probs_arr = np.asarray(probs, dtype=float)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")

    if window_size == 1:
        return probs_arr.copy()

    if probs_arr.ndim == 1:
        return _smooth_1d(probs_arr, window_size, causal=causal)
    elif probs_arr.ndim == 2:
        smoothed = np.zeros_like(probs_arr)
        for c in range(probs_arr.shape[1]):
            smoothed[:, c] = _smooth_1d(probs_arr[:, c], window_size, causal=causal)
        # Re-normalize across classes if multi-class
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return smoothed / row_sums
    else:
        raise ValueError(f"Expected 1D or 2D array, got {probs_arr.ndim}D")


def _smooth_1d(series: np.ndarray, k: int, causal: bool) -> np.ndarray:
    n = len(series)
    if n == 0:
        return series.copy()

    if not causal:
        # Centered median filter with nearest boundary padding
        return median_filter(series, size=k, mode="nearest")

    # Causal (trailing/left-sided) rolling median
    s = pd.Series(series)
    return s.rolling(window=k, min_periods=1).median().to_numpy()


def smooth_labels(
    labels: np.ndarray,
    window_size: int = 3,
    causal: bool = False,
) -> np.ndarray:
    """Apply majority/median filter to discrete integer sleep stages."""
    labels_arr = np.asarray(labels, dtype=int)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")

    if window_size == 1:
        return labels_arr.copy()

    if not causal:
        return median_filter(labels_arr, size=window_size, mode="nearest").astype(int)

    s = pd.Series(labels_arr)
    return s.rolling(window=window_size, min_periods=1).median().round().astype(int).to_numpy()


class TemporalSmoother(BaseEstimator, TransformerMixin):
    """Post-processing estimator that applies temporal smoothing to predictions."""

    def __init__(
        self,
        window_size: int = 3,
        threshold: float = 0.5,
        causal: bool = False,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.causal = causal

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> TemporalSmoother:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            return smooth_labels(arr, window_size=self.window_size, causal=self.causal)
        return smooth_probabilities(arr, window_size=self.window_size, causal=self.causal)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return smooth_probabilities(probs, window_size=self.window_size, causal=self.causal)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        smoothed_probs = self.predict_proba(probs)
        if smoothed_probs.ndim == 1 or (smoothed_probs.ndim == 2 and smoothed_probs.shape[1] == 1):
            p_wake = smoothed_probs.ravel()
            return np.where(p_wake >= self.threshold, 0, 1)
        elif smoothed_probs.ndim == 2 and smoothed_probs.shape[1] == 2:
            p_wake = smoothed_probs[:, 0]
            return np.where(p_wake >= self.threshold, 0, 1)
        else:
            return np.argmax(smoothed_probs, axis=1)


def tune_smoother_on_oof(
    oof_df: pd.DataFrame,
    prob_col: str = "prob_wake_lr_30",
    target_col: str = "y_true_binary",
    threshold: float = 0.5,
    candidate_windows: list[int] | None = None,
    causal: bool = False,
) -> tuple[int, float]:
    """Find optimal smoothing window on training out-of-fold predictions."""
    if candidate_windows is None:
        candidate_windows = [1, 3, 5, 7]

    y_true = oof_df[target_col].to_numpy()

    best_k = 1
    best_f1 = -1.0

    # Group by subject to apply smoothing per recording (prevent cross-subject boundary bleed)
    for k in candidate_windows:
        sm_series = pd.Series(index=oof_df.index, dtype=float)
        for _, grp in oof_df.groupby("subject_id", sort=False):
            sub_p = grp[prob_col].to_numpy()
            sm = smooth_probabilities(sub_p, window_size=k, causal=causal)
            sm_series.loc[grp.index] = sm
        preds = np.where(sm_series.to_numpy() >= threshold, 0, 1)
        score = float(f1_score(y_true, preds, labels=[0], average=None, zero_division=0)[0])
        if score > best_f1:
            best_f1 = score
            best_k = k

    return best_k, best_f1
