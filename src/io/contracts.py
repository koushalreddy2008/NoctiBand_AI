"""
Data contracts and schema representations for NoctiBand AI.

Defines standardized data structures for raw recordings, sleep stages,
signal quality, and epoched data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
import pandas as pd


class SleepStage(IntEnum):
    """Standard sleep stage definitions (AASM standard mapping)."""

    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4
    UNKNOWN = 5
    UNLABELLED = -1

    @classmethod
    def to_name(cls, stage_code: float) -> str:
        """Convert numeric stage code to readable name."""
        if pd.isna(stage_code):
            return "Unlabelled"
        code = int(stage_code)
        mapping = {
            cls.WAKE: "Wake",
            cls.N1: "N1 (Light NREM)",
            cls.N2: "N2 (Core NREM)",
            cls.N3: "N3 (Deep NREM)",
            cls.REM: "REM",
            cls.UNKNOWN: "Unknown/Artifact",
            cls.UNLABELLED: "Unlabelled",
        }
        return mapping.get(code, f"Stage {code}")

    @classmethod
    def to_binary(cls, stage_code: float) -> int | None:
        """Convert 5-stage code to binary (0 = Wake, 1 = Sleep)."""
        if pd.isna(stage_code):
            return None
        code = int(stage_code)
        if code == cls.WAKE:
            return 0
        if code in (cls.N1, cls.N2, cls.N3, cls.REM):
            return 1
        return None

    @classmethod
    def to_3class(cls, stage_code: float) -> int | None:
        """Convert 5-stage code to 3-class (0 = Wake, 1 = NREM, 2 = REM)."""
        if pd.isna(stage_code):
            return None
        code = int(stage_code)
        if code == cls.WAKE:
            return 0
        if code in (cls.N1, cls.N2, cls.N3):
            return 1
        if code == cls.REM:
            return 2
        return None


class SignalQuality(StrEnum):
    """Signal quality classification flags for 30-second epochs."""

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    MISSING_HR = "MISSING_HR"
    MISSING_MOTION = "MISSING_MOTION"
    EMPTY = "EMPTY"


@dataclass
class NightRecording:
    """Standardized representation of an un-epoched overnight recording."""

    subject_id: str
    night_id: str
    rec_start: pd.Timestamp
    epoch_duration_sec: float = 30.0
    hr: pd.DataFrame | None = None
    motion: pd.DataFrame | None = None
    bp: pd.DataFrame | None = None
    spo2: pd.DataFrame | None = None
    labels: np.ndarray | None = None
    dreem_labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hr(self) -> bool:
        """Check if heart rate data is present and non-empty."""
        return self.hr is not None and not self.hr.empty

    @property
    def has_motion(self) -> bool:
        """Check if motion data is present and non-empty."""
        return self.motion is not None and not self.motion.empty

    @property
    def has_labels(self) -> bool:
        """Check if expert sleep stage labels are present."""
        return self.labels is not None and len(self.labels) > 0

    @property
    def has_bp(self) -> bool:
        """Check if blood pressure data is present and non-empty."""
        return self.bp is not None and not self.bp.empty

    @property
    def has_spo2(self) -> bool:
        """Check if SpO2 data is present and non-empty."""
        return self.spo2 is not None and not self.spo2.empty

    @property
    def total_epochs(self) -> int:
        """Total number of epochs. Uses labels length if present, else computes from signals."""
        if self.has_labels and self.labels is not None:
            return len(self.labels)

        max_timestamp = 0.0
        rec_start_unix = self.rec_start.timestamp()
        for df in [self.hr, self.motion, self.bp, self.spo2]:
            if df is not None and not df.empty and "timestamp" in df.columns:
                max_timestamp = max(max_timestamp, float(df["timestamp"].max()))

        if max_timestamp > rec_start_unix:
            return int(np.ceil((max_timestamp - rec_start_unix) / self.epoch_duration_sec))
        return 0

    @property
    def duration_seconds(self) -> float:
        """Total duration of recording in seconds."""
        return float(self.total_epochs * self.epoch_duration_sec)


@dataclass
class EpochedNight:
    """Standardized representation of a 30-second epoched overnight recording."""

    subject_id: str
    night_id: str
    rec_start: pd.Timestamp
    epoch_duration_sec: float
    epochs_df: pd.DataFrame
    raw_slices: dict[int, dict[str, pd.DataFrame]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_epochs(self) -> int:
        """Return total number of aligned epochs."""
        return len(self.epochs_df)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the tabular epoch features and labels."""
        return self.epochs_df.copy()

    def get_epoch_slice(self, epoch_idx: int) -> dict[str, pd.DataFrame]:
        """Get raw signal slices for a specific epoch, if preserved."""
        if self.raw_slices and epoch_idx in self.raw_slices:
            return self.raw_slices[epoch_idx]
        return {}
