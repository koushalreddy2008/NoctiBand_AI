"""
NoctiBand AI - Wearable sleep-monitoring and sleep-improvement support system.
"""

from src.io.contracts import EpochedNight, NightRecording, SignalQuality, SleepStage
from src.io.loader import (
    load_bp,
    load_hr,
    load_labels,
    load_motion,
    load_night,
    load_spo2,
)
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

__version__ = "0.1.0"

__all__ = [
    "EpochAligner",
    "EpochedNight",
    "NightRecording",
    "SignalQuality",
    "SleepStage",
    "align_night_to_epochs",
    "audit_signal_alignment",
    "compute_strided_epoch_aggregations",
    "epoch_to_datetimes",
    "epoch_to_timestamps",
    "filter_and_align_signal",
    "iter_align_night_chunks",
    "load_bp",
    "load_hr",
    "load_labels",
    "load_motion",
    "load_night",
    "load_spo2",
    "timestamp_to_epoch",
]
