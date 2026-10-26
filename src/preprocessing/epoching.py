"""30-second epoching and multimodal signal alignment for NoctiBand AI.

Re-exports from noctiband.preprocessing.epoching for backward compatibility.
"""

from __future__ import annotations

from noctiband.preprocessing.epoching import (
    EpochAligner,
    align_night_to_epochs,
    assign_epochs,
    compute_epoch_coverage,
    compute_strided_epoch_aggregations,
    iter_align_night_chunks,
    summarize_uniform_signal_window,
)

__all__ = [
    "EpochAligner",
    "align_night_to_epochs",
    "assign_epochs",
    "compute_epoch_coverage",
    "compute_strided_epoch_aggregations",
    "iter_align_night_chunks",
    "summarize_uniform_signal_window",
]
