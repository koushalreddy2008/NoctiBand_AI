"""Estimator registry and model construction for sleep stage models."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from noctiband.models.baselines import MajorityBaseline, MotionThresholdBaseline

SEED = 42

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


def build_model(kind: str, params: dict | None = None) -> Pipeline:
    """Build a leak-free sklearn Pipeline containing imputer, scaler, and model."""
    params = dict(params or {})
    if kind == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        clf = LogisticRegression(random_state=SEED, **params)
    elif kind == "random_forest":
        clf = RandomForestClassifier(random_state=SEED, **params)
    elif kind == "svm":
        clf = SVC(probability=True, random_state=SEED, **params)
    elif kind in ("xgboost", "xgb"):
        from xgboost import XGBClassifier

        xgb_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": SEED,
        }
        xgb_params.update(params)
        clf = XGBClassifier(**xgb_params)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


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
register_estimator("svm", lambda: SVC(probability=True, random_state=42))


def _make_xgb_estimator() -> BaseEstimator:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


register_estimator("xgboost", _make_xgb_estimator)
register_estimator("xgb", _make_xgb_estimator)

