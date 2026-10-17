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
)

__all__ = [
    "EpochAligner",
    "align_night_to_epochs",
    "audit_signal_alignment",
    "epoch_to_datetimes",
    "epoch_to_timestamps",
    "filter_and_align_signal",
    "timestamp_to_epoch",
]
