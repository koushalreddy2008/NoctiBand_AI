"""I/O module for NoctiBand AI.

Provides data contracts and loaders for physiological signals, motion, and sleep labels.
"""

from __future__ import annotations

from noctiband.io.contracts import EpochedNight, NightRecording, SignalQuality, SleepStage
from noctiband.io.loader import (
    is_header_present,
    load_acceleration,
    load_bp,
    load_hr,
    load_labels,
    load_motion,
    load_spo2,
    parse_rec_start,
)
from noctiband.io.loader import (
    load_night_recording as load_night,
)

__all__ = [
    "EpochedNight",
    "NightRecording",
    "SignalQuality",
    "SleepStage",
    "is_header_present",
    "load_acceleration",
    "load_bp",
    "load_hr",
    "load_labels",
    "load_motion",
    "load_night",
    "load_spo2",
    "parse_rec_start",
]
