"""Evaluation harness and metrics for NoctiBand AI."""

from noctiband.evaluation.harness import (
    EvaluationResult,
    build_pipeline,
    compute_metrics,
    evaluate_cv,
)

__all__ = [
    "EvaluationResult",
    "build_pipeline",
    "compute_metrics",
    "evaluate_cv",
]
