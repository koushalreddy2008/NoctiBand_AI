"""Inference module for NoctiBand."""

from __future__ import annotations

from noctiband.inference.predict import load_inference_model, predict_night

__all__ = ["predict_night", "load_inference_model"]
