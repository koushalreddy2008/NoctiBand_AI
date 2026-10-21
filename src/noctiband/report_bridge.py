"""Report bridge and sleep architecture metric calculation (Person A / Person B bridge).

Calculates standard polysomnography and actigraphy metrics from NightPrediction,
including TST, Sleep Efficiency, WASO, SOL, and NOA, and attaches non-diagnostic
sleep hygiene recommendations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from noctiband.types import NightPrediction

DISCLAIMER = (
    "NoctiBand provides consumer wellness metrics and sleep architecture insights. "
    "It is not a medical device and is not intended to diagnose, treat, cure, or prevent "
    "any disease or medical sleep disorder."
)


def compute_sleep_metrics(epoch_labels: np.ndarray) -> dict[str, Any]:
    """Compute standard actigraphic sleep metrics from 30-second epoch labels.

    Parameters
    ----------
    epoch_labels : np.ndarray
        1D array of binary sleep stages (0: Wake, 1: Sleep).

    Returns
    -------
    dict[str, Any]
        Dictionary with TST, SE, WASO, SOL, NOA, and window totals.
    """
    labels = np.asarray(epoch_labels, dtype=int)
    n_epochs = len(labels)

    if n_epochs == 0:
        return {
            "total_window_min": 0.0,
            "tst_min": 0.0,
            "waso_min": 0.0,
            "sol_min": 0.0,
            "sleep_efficiency": 0.0,
            "noa": 0,
            "sleep_epochs": 0,
            "wake_epochs": 0,
        }

    total_window_min = round(n_epochs * 0.5, 2)  # 30 sec = 0.5 min
    sleep_indices = np.where(labels == 1)[0]

    if len(sleep_indices) == 0:
        # All wake night
        return {
            "total_window_min": total_window_min,
            "tst_min": 0.0,
            "waso_min": 0.0,
            "sol_min": total_window_min,
            "sleep_efficiency": 0.0,
            "noa": 0,
            "sleep_epochs": 0,
            "wake_epochs": n_epochs,
        }

    sleep_onset_idx = int(sleep_indices[0])
    final_awakening_idx = int(sleep_indices[-1])

    sol_min = round(sleep_onset_idx * 0.5, 2)
    tst_min = round(len(sleep_indices) * 0.5, 2)

    # Wake After Sleep Onset: wake epochs between sleep onset and final awakening
    in_sleep_window = labels[sleep_onset_idx : final_awakening_idx + 1]
    waso_epochs = int((in_sleep_window == 0).sum())
    waso_min = round(waso_epochs * 0.5, 2)

    sleep_efficiency = round((tst_min / total_window_min) * 100.0, 2)

    # Number of Awakenings (transitions from 1 -> 0 within the sleep period)
    diffs = np.diff(in_sleep_window)
    noa = int((diffs == -1).sum())

    return {
        "total_window_min": total_window_min,
        "tst_min": tst_min,
        "waso_min": waso_min,
        "sol_min": sol_min,
        "sleep_efficiency": sleep_efficiency,
        "noa": noa,
        "sleep_epochs": int(len(sleep_indices)),
        "wake_epochs": int((labels == 0).sum()),
        "sleep_onset_epoch": sleep_onset_idx,
        "final_awakening_epoch": final_awakening_idx,
    }


def generate_hygiene_recommendations(metrics: dict[str, Any]) -> list[str]:
    """Map measured sleep metrics to evidence-based, non-diagnostic hygiene tips."""
    tips: list[str] = []
    se = metrics.get("sleep_efficiency", 0.0)
    waso = metrics.get("waso_min", 0.0)
    sol = metrics.get("sol_min", 0.0)

    if se < 85.0:
        tips.append(
            "Maintain a consistent wake time every morning, even on weekends, "
            "to strengthen circadian rhythm and improve sleep consolidation."
        )

    if waso > 40.0:
        tips.append(
            "To reduce mid-night awakenings, moderate fluid and caffeine intake in "
            "the 4–6 hours prior to bed and keep the bedroom quiet, dark, and cool (65–68°F)."
        )

    if sol > 30.0:
        tips.append(
            "If taking over 20–30 minutes to fall asleep, avoid bright screens before bed "
            "and practice a relaxing wind-down routine (e.g. reading, meditation) in dim lighting."
        )

    if not tips:
        tips.append(
            "Your sleep consolidation and timing are in an optimal range! "
            "Continue maintaining your consistent daily bedtime routine."
        )

    return tips


def make_report(
    prediction: NightPrediction | dict[str, Any],
    subject_id: str = "Subject",
    night_id: str = "Night 1",
) -> dict[str, Any]:
    """Generate a comprehensive sleep metrics report dictionary from NightPrediction.

    Parameters
    ----------
    prediction : NightPrediction or dict
        Predicted night container or dict with 'epoch_labels' / 'hypnogram'.
    subject_id : str, default='Subject'
        Subject or user identifier.
    night_id : str, default='Night 1'
        Night label or date.

    Returns
    -------
    dict[str, Any]
        Complete structured report ready for Streamlit UI or downstream consumers.
    """
    if isinstance(prediction, NightPrediction):
        labels = prediction.epoch_labels
        rec_start = prediction.rec_start
        qc_summary = prediction.qc_summary
        cov = prediction.coverage
    elif isinstance(prediction, dict):
        labels = prediction.get("epoch_labels", prediction.get("hypnogram"))
        rec_start = prediction.get("rec_start", 0.0)
        qc_summary = prediction.get("qc_summary", {})
        cov = prediction.get("coverage", None)
    else:
        raise TypeError(f"Expected NightPrediction or dict, got {type(prediction)}")

    metrics = compute_sleep_metrics(labels)
    recommendations = generate_hygiene_recommendations(metrics)

    return {
        "subject_id": subject_id,
        "night_id": night_id,
        "rec_start": rec_start,
        "metrics": metrics,
        "hypnogram": labels.tolist() if isinstance(labels, np.ndarray) else list(labels),
        "coverage": cov.tolist() if isinstance(cov, np.ndarray) else cov,
        "qc_summary": qc_summary,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER,
    }
