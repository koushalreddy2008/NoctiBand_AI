import numpy as np

from noctiband.types import EPOCH_S, STAGE_NAMES, NightPrediction


def test_prediction_shapes():
    p = NightPrediction(0.0, np.zeros(10, dtype=int), None, np.ones(10))
    assert EPOCH_S == 30
    assert p.epoch_labels.shape == p.coverage.shape


def test_stage_coding():
    assert STAGE_NAMES[0] == "Wake" and len(STAGE_NAMES) == 5