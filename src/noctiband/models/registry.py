"""Estimator registry for sleep stage models.

Allows decoupled registration of machine learning models (including external
models like SVM from Person B) without modifying the evaluation harness.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from noctiband.models.baselines import MajorityBaseline, MotionThresholdBaseline

_REGISTRY: dict[str, Callable[[], BaseEstimator]] = {}


def register_estimator(name: str, factory: Callable[[], BaseEstimator]) -> None:
    """Register a new estimator factory by unique name.

    Parameters
    ----------
    name : str
        Unique identifier for the estimator.
    factory : Callable[[], BaseEstimator]
        Callable returning a fresh instance of a scikit-learn BaseEstimator.
    """
    _REGISTRY[name.lower()] = factory


def get_estimator(name: str) -> BaseEstimator:
    """Instantiate and return an estimator registered under name.

    Parameters
    ----------
    name : str
        Registered name of the estimator.

    Returns
    -------
    BaseEstimator
        Fresh instance of the requested estimator.

    Raises
    ------
    KeyError
        If name is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Estimator '{name}' not found. Available estimators: {available}")
    return _REGISTRY[key]()


def list_estimators() -> list[str]:
    """Return a sorted list of all registered estimator names."""
    return sorted(_REGISTRY.keys())


# Pre-register built-in models
register_estimator("majority", lambda: MajorityBaseline())
register_estimator("motion_threshold", lambda: MotionThresholdBaseline())
register_estimator(
    "logistic_regression",
    lambda: LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
)
register_estimator(
    "random_forest",
    lambda: RandomForestClassifier(
        class_weight="balanced_subsample", random_state=42, n_estimators=100
    ),
)
