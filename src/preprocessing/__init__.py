"""
Preprocessing module for NoctiBand AI.

Provides timestamp alignment, 30-second epoching, signal quality auditing,
and multimodal feature synchronization.
"""

from src.preprocessing.alignment import (
    audit_signal_alignment,
    epoch_to_datetimes,
    epoch_to_timestamps,
    filter_and_align_signal,
    timestamp_to_epoch,
)
from src.preprocessing.epoching import (
    EpochAligner,
    align_night_to_epochs,
    compute_strided_epoch_aggregations,
    iter_align_night_chunks,
)

__all__ = [
    "EpochAligner",
    "align_night_to_epochs",
    "audit_signal_alignment",
    "compute_strided_epoch_aggregations",
    "epoch_to_datetimes",
    "epoch_to_timestamps",
    "filter_and_align_signal",
    "iter_align_night_chunks",
    "timestamp_to_epoch",
]
