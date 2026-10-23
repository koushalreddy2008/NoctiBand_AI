"""Model Manifest Schema and Validation Engine for NoctiBand AI (v0.2).

Ensures versioned model artifacts strictly match feature dimensions,
sampling rate specifications, and calibrated decision thresholds prior
to inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelManifest:
    """Versioned specification and validation metadata for a trained model artifact."""

    model_name: str
    version: str
    architecture: str
    features: list[str]
    n_features: int
    optimal_wake_threshold: float
    smoothing_window_size: int = 1
    target_mapping: dict[str, str] = field(default_factory=dict)
    min_sampling_rate_hz: float = 10.0
    max_sampling_rate_hz: float = 100.0
    training_cohort: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    def validate(self) -> None:
        """Validate structural integrity and physical bounds of the manifest.

        Raises
        ------
        ValueError
            If feature length mismatch, invalid threshold, or invalid sample rate bounds.
        """
        if len(self.features) != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, "
                f"got {len(self.features)} features in manifest."
            )

        if not (0.0 <= self.optimal_wake_threshold <= 1.0):
            raise ValueError(
                f"Optimal threshold {self.optimal_wake_threshold} outside valid range [0.0, 1.0]."
            )

        if self.min_sampling_rate_hz <= 0.0:
            raise ValueError(
                f"Minimum sampling rate must be positive, got {self.min_sampling_rate_hz} Hz."
            )

        if self.max_sampling_rate_hz < self.min_sampling_rate_hz:
            raise ValueError(
                f"Max sampling rate ({self.max_sampling_rate_hz}) cannot be less than "
                f"min sampling rate ({self.min_sampling_rate_hz})."
            )

        if self.smoothing_window_size < 1 or self.smoothing_window_size % 2 == 0:
            raise ValueError(
                f"Smoothing window size must be an odd positive integer, "
                f"got {self.smoothing_window_size}."
            )

    def validate_features(self, input_features: list[str] | set[str] | Any) -> bool:
        """Verify that all required manifest features exist in the input feature space.

        Parameters
        ----------
        input_features : collection of str
            Available feature names provided to the inference engine.

        Returns
        -------
        bool
            True if all required features exist.

        Raises
        ------
        ValueError
            If any required features are absent.
        """
        available = set(input_features)
        missing = [f for f in self.features if f not in available]
        if missing:
            raise ValueError(
                f"Input data is missing {len(missing)} required model features: {missing}"
            )
        return True

    def validate_sampling_rate(self, sampling_rate_hz: float) -> bool:
        """Verify sensor sampling rate is within validated device bounds.

        Parameters
        ----------
        sampling_rate_hz : float
            Empirical sensor stream sampling rate.

        Returns
        -------
        bool
            True if within acceptable rate range.

        Raises
        ------
        ValueError
            If sampling rate is outside compatible bounds.
        """
        if not (self.min_sampling_rate_hz <= sampling_rate_hz <= self.max_sampling_rate_hz):
            raise ValueError(
                f"Sampling rate {sampling_rate_hz:.2f} Hz is outside validated bounds "
                f"[{self.min_sampling_rate_hz}, {self.max_sampling_rate_hz}] Hz."
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "architecture": self.architecture,
            "features": list(self.features),
            "n_features": self.n_features,
            "optimal_wake_threshold": self.optimal_wake_threshold,
            "smoothing_window_size": self.smoothing_window_size,
            "target_mapping": dict(self.target_mapping),
            "min_sampling_rate_hz": self.min_sampling_rate_hz,
            "max_sampling_rate_hz": self.max_sampling_rate_hz,
            "training_cohort": dict(self.training_cohort),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        """Construct and validate ModelManifest from dictionary."""
        manifest = cls(
            model_name=data.get("model_name", "Unknown Model"),
            version=data.get("version", "v0.0"),
            architecture=data.get("architecture", "Pipeline"),
            features=list(data.get("features", [])),
            n_features=int(data.get("n_features", len(data.get("features", [])))),
            optimal_wake_threshold=float(data.get("optimal_wake_threshold", 0.5)),
            smoothing_window_size=int(data.get("smoothing_window_size", 1)),
            target_mapping=dict(data.get("target_mapping", {})),
            min_sampling_rate_hz=float(data.get("min_sampling_rate_hz", 10.0)),
            max_sampling_rate_hz=float(data.get("max_sampling_rate_hz", 100.0)),
            training_cohort=dict(data.get("training_cohort", {})),
            checksum=data.get("checksum"),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_file(cls, path: str | Path) -> ModelManifest:
        """Load and validate manifest from JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model manifest file not found: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
