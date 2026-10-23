"""Unit tests for ModelManifest schema validation and checking."""

from pathlib import Path

import pytest

from noctiband.models.manifest import ModelManifest


def test_manifest_validation_success():
    m = ModelManifest(
        model_name="Test Model",
        version="v0.2",
        architecture="StandardScaler -> LogisticRegression",
        features=["hr_mean", "acc_mag_mean"],
        n_features=2,
        optimal_wake_threshold=0.75,
        smoothing_window_size=3,
        min_sampling_rate_hz=10.0,
        max_sampling_rate_hz=100.0,
    )
    m.validate()
    assert m.validate_features(["hr_mean", "acc_mag_mean", "extra_feat"])
    assert m.validate_sampling_rate(50.0)


def test_manifest_dimension_mismatch():
    m = ModelManifest(
        model_name="Invalid Model",
        version="v0.2",
        architecture="Pipeline",
        features=["hr_mean"],
        n_features=2,
        optimal_wake_threshold=0.5,
    )
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        m.validate()


def test_manifest_sampling_rate_bounds():
    m = ModelManifest(
        model_name="Rate Bound Model",
        version="v0.2",
        architecture="Pipeline",
        features=["hr_mean"],
        n_features=1,
        optimal_wake_threshold=0.5,
        min_sampling_rate_hz=10.0,
        max_sampling_rate_hz=60.0,
    )
    m.validate()
    assert m.validate_sampling_rate(25.0)
    with pytest.raises(ValueError, match="outside validated bounds"):
        m.validate_sampling_rate(5.0)


def test_manifest_from_existing_v01_file():
    manifest_path = Path("models/noctiband_v0.1_meta.json")
    if manifest_path.exists():
        manifest = ModelManifest.from_file(manifest_path)
        assert manifest.n_features == 30
        assert manifest.optimal_wake_threshold == 0.85
        assert len(manifest.features) == 30
