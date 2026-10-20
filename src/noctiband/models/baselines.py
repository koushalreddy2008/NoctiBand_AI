"""Baseline models for 30-second epoch sleep stage classification."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Majority class baseline classifier.

    Learns the most frequent label in training data and constantly predicts it.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        self.majority_class_ = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X):
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        proba = np.zeros((n_samples, len(self.classes_)), dtype=float)
        maj_idx = np.where(self.classes_ == self.majority_class_)[0][0]
        proba[:, maj_idx] = 1.0
        return proba


class MotionThresholdBaseline(BaseEstimator, ClassifierMixin):
    """Motion threshold rule baseline classifier.

    Learns a threshold strictly from training folds to classify Wake vs Sleep.
    Higher motion indicates Wake (0); lower motion indicates Sleep (1).
    For multi-class, predicts Wake (0) when above threshold and NREM (1) below.
    """

    def __init__(self, feature_idx: int | None = None):
        self.feature_idx = feature_idx

    def _get_effective_idx(self, n_features: int) -> int:
        if self.feature_idx is not None:
            return min(self.feature_idx, n_features - 1)
        # In MODEL_FEATURES, motion_mag_mean is at index 8; fallback to 0 for small fixtures
        return 8 if n_features > 8 else 0

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        idx = self._get_effective_idx(X_arr.shape[1])
        self.effective_idx_ = idx

        feature_vals = X_arr[:, idx]
        finite_mask = np.isfinite(feature_vals)
        if not finite_mask.any():
            self.threshold_ = 0.0
            return self

        valid_vals = feature_vals[finite_mask]
        valid_y = y_arr[finite_mask]

        # Candidate thresholds from quantiles
        quantiles = np.linspace(0.01, 0.99, 100)
        candidates = np.quantile(valid_vals, quantiles)

        best_score = -1.0
        best_thresh = float(np.median(valid_vals))

        for thresh in candidates:
            # Predict Wake (0) if feature > thresh, else Sleep (1)
            preds = np.where(valid_vals > thresh, 0, 1)
            score = balanced_accuracy_score(valid_y, preds)
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)

        self.threshold_ = best_thresh
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        idx = getattr(self, "effective_idx_", self._get_effective_idx(X_arr.shape[1]))
        vals = X_arr[:, idx]
        # Predict Wake (0) if > threshold, else Sleep (1)
        preds = np.where(vals > self.threshold_, 0, 1)

        # Ensure prediction matches classes present
        if len(self.classes_) > 2:
            # 3-class fallback: Wake (0) vs NREM (1)
            preds = np.where(vals > self.threshold_, 0, 1)

        return preds

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        idx = getattr(self, "effective_idx_", self._get_effective_idx(X_arr.shape[1]))
        vals = X_arr[:, idx]
        diff = vals - self.threshold_
        # Logistic sigmoid probability of Wake (class 0)
        p_wake = 1.0 / (1.0 + np.exp(-np.clip(diff, -10.0, 10.0)))

        if len(self.classes_) == 2 and 0 in self.classes_ and 1 in self.classes_:
            proba = np.column_stack([p_wake, 1.0 - p_wake])
        else:
            proba = np.zeros((len(X_arr), len(self.classes_)))
            if 0 in self.classes_:
                w_idx = np.where(self.classes_ == 0)[0][0]
                proba[:, w_idx] = p_wake
                rem = (1.0 - p_wake) / max(1, len(self.classes_) - 1)
                for i in range(len(self.classes_)):
                    if i != w_idx:
                        proba[:, i] = rem
            else:
                proba[:, 0] = 1.0
        return proba
