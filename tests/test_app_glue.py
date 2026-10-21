"""Unit tests for Streamlit backend glue and multi-night aggregation (Stage 4)."""

import io

import numpy as np
import pytest

from noctiband.app_glue import (
    aggregate_multi_night,
    get_cached_model,
    load_and_validate_upload,
)
from noctiband.types import NightData, NightPrediction


def test_load_and_validate_upload_valid_dict():
    # Valid CSV buffers
    hr_csv = "timestamp,hr\n0.0,60.0\n1.0,62.0\n2.0,61.0\n"
    mot_csv = "timestamp,x,y,z\n0.0,0.0,0.0,1.0\n0.02,0.01,0.0,0.99\n"

    files = {
        "hr": io.StringIO(hr_csv),
        "motion": io.StringIO(mot_csv),
    }

    night = load_and_validate_upload(files, subject_id="user_123")
    assert isinstance(night, NightData)
    assert night.subject == "user_123"
    assert len(night.hr) == 3
    assert len(night.motion) == 2
    assert night.rec_start == 0.0


def test_load_and_validate_upload_corrupt_files():
    # Corrupt empty HR
    files_empty = {
        "hr": io.StringIO(""),
        "motion": io.StringIO("timestamp,x,y,z\n0.0,0,0,1\n"),
    }
    with pytest.raises(ValueError, match="empty|Missing"):
        load_and_validate_upload(files_empty)

    # Missing motion file
    files_missing = {"hr": io.StringIO("timestamp,hr\n0,60\n")}
    with pytest.raises(ValueError, match="Missing accelerometer"):
        load_and_validate_upload(files_missing)

    # Non-numeric garbage
    files_garbage = {
        "hr": io.StringIO("timestamp,hr\ngarbage,text\n"),
        "motion": io.StringIO("timestamp,x,y,z\n0.0,0,0,1\n"),
    }
    with pytest.raises(ValueError, match="valid numeric"):
        load_and_validate_upload(files_garbage)


def test_aggregate_multi_night_math():
    # Create 3 synthetic nights with known sleep efficiencies
    # Night 1: 10 epochs, all sleep (1) -> SE = 100%
    n1 = NightPrediction(
        rec_start=0.0,
        epoch_labels=np.ones(10, dtype=int),
        probs=np.ones(10),
        coverage=np.ones(10),
    )
    # Night 2: 10 epochs, 5 wake (0) then 5 sleep (1) -> SE = 50%
    n2 = NightPrediction(
        rec_start=0.0,
        epoch_labels=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        probs=np.ones(10),
        coverage=np.ones(10),
    )
    # Night 3: 10 epochs, all wake (0) -> SE = 0%
    n3 = NightPrediction(
        rec_start=0.0,
        epoch_labels=np.zeros(10, dtype=int),
        probs=np.zeros(10),
        coverage=np.ones(10),
    )

    agg = aggregate_multi_night([n1, n2, n3])
    assert agg["n_nights"] == 3
    # Mean SE = (100 + 50 + 0) / 3 = 50.0%
    assert agg["mean_sleep_efficiency"] == 50.0
    # Slope of [100, 50, 0] over x=[0, 1, 2] is exactly -50.0% / night
    assert np.isclose(agg["efficiency_trend_slope"], -50.0, atol=1e-3)
    # Verify SRI is calculated
    assert 0.0 <= agg["sleep_regularity_index"] <= 100.0


def test_cached_model_loader_idempotent():
    model_obj1 = get_cached_model()
    model_obj2 = get_cached_model()
    assert model_obj1 is model_obj2  # Same reference / cached
    assert "pipeline" in model_obj1
    assert "optimal_threshold" in model_obj1
