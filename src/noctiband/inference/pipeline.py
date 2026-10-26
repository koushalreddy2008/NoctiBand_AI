"""
End-to-end night prediction and inference pipeline for NoctiBand AI (Person A).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from noctiband.io.loader import load_night
from noctiband.models.data import ROOT
from noctiband.postprocess.smoothing import median_smooth
from noctiband.preprocessing.epoching import align_night_to_epochs


def run_night(
    night_folder: str | Path,
    model_name: str = "logreg",
    window: int = 3,
) -> pd.DataFrame:
    """Run end-to-end sleep stage prediction for an overnight recording folder.

    Parameters
    ----------
    night_folder : str or Path
        Directory path containing raw overnight channels (hr.csv, motion.csv, labels.mat).
    model_name : str, default="logreg"
        Model architecture to load from artifacts/models/{model_name}.joblib.
    window : int, default=3
        Temporal smoothing window size for median filtering.

    Returns
    -------
    pd.DataFrame
        One row per 30-second epoch with columns:
        - 'epoch_idx': zero-based epoch index
        - 'y_pred': binary prediction (0 = Wake, 1 = Sleep)
        - 'p_sleep': smoothed sleep probability
        - 'quality_flag': epoch signal quality flag
    """
    path = Path(night_folder)
    if not path.exists():
        # Fallback check for default data/raw if specific subpath does not exist
        alt = ROOT / "data" / "raw"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Recording folder not found: {night_folder}")

    # 1. Load raw recording session
    rec = load_night(path, allow_partial=True)

    # 2. Segment and align signals into 30-second epochs
    epoched = align_night_to_epochs(rec)
    df = epoched.epochs_df.copy()
    if len(df) == 0:
        raise ValueError(f"Recording at {night_folder} produced 0 epochs.")

    # 3. Construct canonical 16-feature space matching model contracts
    df["acc_vm_mean"] = df["vm_mean"] if "vm_mean" in df.columns else 0.0
    df["acc_vm_std"] = df["vm_std"] if "vm_std" in df.columns else 0.0
    df["acc_vm_max"] = df["vm_max"] if "vm_max" in df.columns else 0.0
    df["acc_enmo_mean"] = df["enmo_mean"] if "enmo_mean" in df.columns else 0.0
    df["ctx_hour_of_night"] = (df["epoch"] * 30.0) / 3600.0
    df["q_hr_count"] = df["hr_count"] if "hr_count" in df.columns else 0
    df["q_motion_count"] = df["motion_count"] if "motion_count" in df.columns else 0

    # Ensure all feature columns exist with numeric dtypes
    model_dir = ROOT / "artifacts" / "models"
    feat_path = model_dir / f"{model_name}_features.json"
    model_path = model_dir / f"{model_name}.joblib"

    if not model_path.exists() or not feat_path.exists():
        raise FileNotFoundError(
            f"Trained model artifacts not found for '{model_name}'. "
            f"Expected {model_path} and {feat_path}."
        )

    with open(feat_path) as f:
        feats = json.load(f)["features"]

    for col in feats:
        if col not in df.columns:
            df[col] = 0.0

    # 4. Predict sleep probabilities
    model = joblib.load(model_path)
    classes = list(model.classes_)
    sleep_idx = classes.index(1) if 1 in classes else 0
    probs = model.predict_proba(df[feats])
    p_sleep = probs[:, sleep_idx]

    # 5. Apply tuned temporal median smoothing
    p_smooth = median_smooth(p_sleep, window=window)
    y_pred = (p_smooth >= 0.5).astype(int)

    # 6. Build standardized prediction DataFrame
    pred_df = pd.DataFrame({
        "epoch_idx": df["epoch"].astype(int),
        "y_pred": y_pred,
        "p_sleep": p_smooth,
        "quality_flag": df["quality_flag"] if "quality_flag" in df.columns else "GOOD",
    })

    # Preserve recording start timestamp in DataFrame metadata
    if rec.rec_start is not None:
        pred_df.attrs["rec_start"] = rec.rec_start

    return pred_df
