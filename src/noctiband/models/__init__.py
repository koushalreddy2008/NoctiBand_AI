"""Models package for NoctiBand AI."""

from __future__ import annotations

from noctiband.models.baselines import MajorityBaseline, MotionThresholdBaseline
from noctiband.models.registry import (
    build_model,
    get_estimator,
    list_estimators,
    register_estimator,
)

__all__ = [
    "MajorityBaseline",
    "MotionThresholdBaseline",
    "build_model",
    "get_estimator",
    "list_estimators",
    "register_estimator",
]
