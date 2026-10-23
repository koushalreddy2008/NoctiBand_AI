"""
Tests for model registry and configuration files.
"""

from pathlib import Path

import pytest
import yaml

import src.models.registry as legacy_registry
from noctiband.models.registry import SEED, build_model

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "models"


def test_yaml_configs_exist_and_valid():
    expected_configs = ["dummy.yaml", "logreg.yaml", "rf.yaml", "svm.yaml"]
    for config_name in expected_configs:
        config_path = CONFIGS_DIR / config_name
        assert config_path.is_file(), f"Missing config file: {config_name}"

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        assert "name" in data
        assert "model" in data
        assert "params" in data
        assert isinstance(data["params"], dict)


def test_build_model_dummy():
    try:
        pipe = build_model("dummy")
    except ImportError as e:
        if "Application Control policy" in str(e):
            pytest.skip("Skipping build_model on Windows due to Smart App Control blocking sklearn native DLL")
        raise
    assert "impute" in pipe.named_steps
    assert "scale" in pipe.named_steps
    assert "clf" in pipe.named_steps
    assert pipe.named_steps["clf"].strategy == "most_frequent"


def test_build_model_unknown_kind():
    with pytest.raises(ValueError, match="Unknown model kind: neural_net"):
        build_model("neural_net")


def test_legacy_module_exports_match():
    assert legacy_registry.SEED == SEED
    assert legacy_registry.build_model is build_model
