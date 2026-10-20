"""Models package for NoctiBand AI."""

from noctiband.models.baselines import MajorityBaseline, MotionThresholdBaseline
from noctiband.models.registry import (
    get_estimator,
    list_estimators,
    register_estimator,
)

__all__ = [
    "MajorityBaseline",
    "MotionThresholdBaseline",
    "get_estimator",
    "list_estimators",
    "register_estimator",
]
