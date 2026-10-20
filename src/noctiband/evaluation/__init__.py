"""Evaluation module under noctiband namespace."""

from __future__ import annotations

from noctiband.evaluation.harness import (
    EvaluationResult,
    build_pipeline,
    compute_metrics,
    evaluate_cv,
)
from src.evaluation.metrics import DEFAULT_LABELS, DEFAULT_NAMES, evaluate

__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_NAMES",
    "EvaluationResult",
    "build_pipeline",
    "compute_metrics",
    "evaluate",
    "evaluate_cv",
]
