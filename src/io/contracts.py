"""Data contracts and schema representations for NoctiBand AI.

Re-exports contracts from noctiband.io.contracts for backward compatibility.
"""

from __future__ import annotations

from noctiband.io.contracts import EpochedNight, NightRecording, SignalQuality, SleepStage

__all__ = ["EpochedNight", "NightRecording", "SignalQuality", "SleepStage"]
