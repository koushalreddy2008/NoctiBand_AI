"""Models package for NoctiBand AI."""

from __future__ import annotations

from noctiband.models.baselines import MajorityBaseline, MotionThresholdBaseline
from noctiband.models.manifest import ModelManifest
from noctiband.models.registry import (
    build_model,
    get_estimator,
    list_estimators,
    register_estimator,
)
from noctiband.models.xgboost_model import (
    build_xgboost_pipeline,
    cross_validate_xgboost,
    save_xgboost_artifact,
    train_xgboost,
)

__all__ = [
    "MajorityBaseline",
    "ModelManifest",
    "MotionThresholdBaseline",
    "build_model",
    "build_xgboost_pipeline",
    "cross_validate_xgboost",
    "get_estimator",
    "list_estimators",
    "register_estimator",
    "save_xgboost_artifact",
    "train_xgboost",
]
