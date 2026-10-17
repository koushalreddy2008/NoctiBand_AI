from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPOCH_S = 30  # seconds per epoch; one label per epoch

# BIDSleep expert_label coding (verified in docs/data_notes.md)
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


@dataclass(frozen=True)
class NightData:
    subject: str
    night: str
    hr: pd.DataFrame            # timestamp (unix s), hr (bpm)
    motion: pd.DataFrame        # timestamp (unix s), x, y, z (g)
    labels: np.ndarray | None   # per-epoch stage codes; None for unlabeled band data
    rec_start: float            # unix s at the start of epoch 0


@dataclass(frozen=True)
class NightPrediction:
    rec_start: float
    epoch_labels: np.ndarray    # internal coding: 0 = wake, 1 = sleep
    probs: np.ndarray | None    # P(sleep) per epoch
    coverage: np.ndarray        # fraction of the epoch with valid HR/motion, 0..1