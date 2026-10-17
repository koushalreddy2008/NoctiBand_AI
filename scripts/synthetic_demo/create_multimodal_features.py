# SYNTHETIC DEMO ONLY. Signals are generated from ground-truth sleep labels.
# They leak the target. NEVER use them for training or evaluation.
"""
Script: create_multimodal_features.py
Purpose: Aggregate heart rate, blood pressure, and SpO2 into a synchronized
         30-second epoch feature dataset for machine learning and reporting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

def extract_features():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "synthetic_demo"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labels & start time
    mat = loadmat(raw_dir / "labels.mat")
    expert_labels = mat["expert_label"].ravel()
    rec_start_text = str(mat["recStart"][0])
    rec_start = pd.Timestamp(rec_start_text).tz_localize("America/New_York")
    rec_start_unix = rec_start.timestamp()
    total_epochs = len(expert_labels)

    # 2. Load raw signals
    hr_df = pd.read_csv(raw_dir / "hr.csv", header=None, names=["timestamp", "heart_rate"])
    bp_df = pd.read_csv(raw_dir / "blood_pressure.csv")
    spo2_df = pd.read_csv(raw_dir / "spo2.csv")

    # Map timestamps to 30-second epoch IDs
    hr_df["epoch"] = np.floor((hr_df["timestamp"] - rec_start_unix) / 30.0).astype(int)
    bp_df["epoch"] = np.floor((bp_df["timestamp"] - rec_start_unix) / 30.0).astype(int)
    spo2_df["epoch"] = np.floor((spo2_df["timestamp"] - rec_start_unix) / 30.0).astype(int)

    # Filter to valid recording epochs [0, total_epochs - 1]
    valid_hr = hr_df.loc[hr_df["epoch"].between(0, total_epochs - 1)]
    valid_bp = bp_df.loc[bp_df["epoch"].between(0, total_epochs - 1)]
    valid_spo2 = spo2_df.loc[spo2_df["epoch"].between(0, total_epochs - 1)]

    # 3. Aggregate HR features
    hr_feat = valid_hr.groupby("epoch").agg(
        hr_count=("heart_rate", "count"),
        hr_mean=("heart_rate", "mean"),
        hr_std=("heart_rate", "std"),
        hr_min=("heart_rate", "min"),
        hr_max=("heart_rate", "max"),
    )
    hr_feat["hr_range"] = hr_feat["hr_max"] - hr_feat["hr_min"]

    # 4. Aggregate Blood Pressure features
    bp_feat = valid_bp.groupby("epoch").agg(
        sbp_mean=("systolic", "mean"),
        sbp_std=("systolic", "std"),
        sbp_min=("systolic", "min"),
        sbp_max=("systolic", "max"),
        dbp_mean=("diastolic", "mean"),
        dbp_std=("diastolic", "std"),
        map_mean=("map", "mean"),
        pulse_pressure_mean=("pulse_pressure", "mean")
    )

    # 5. Aggregate SpO2 features
    # Count seconds with SpO2 < 95% within each epoch
    valid_spo2_copy = valid_spo2.copy()
    valid_spo2_copy["low_spo2"] = (valid_spo2_copy["spo2"] < 95.0).astype(int)
    
    spo2_feat = valid_spo2_copy.groupby("epoch").agg(
        spo2_mean=("spo2", "mean"),
        spo2_min=("spo2", "min"),
        spo2_std=("spo2", "std"),
        spo2_sec_below_95=("low_spo2", "sum")
    )

    # 6. Combine all features across all epochs [0, total_epochs - 1]
    combined = pd.DataFrame(index=pd.RangeIndex(total_epochs, name="epoch"))
    combined = combined.join(hr_feat).join(bp_feat).join(spo2_feat).reset_index()

    # Add sleep stage label
    combined["sleep_stage"] = expert_labels

    output_path = processed_dir / "multimodal_epoch_features.csv"
    combined.to_csv(output_path, index=False)
    print(f"Created multimodal feature table at: {output_path}")
    print(f"Shape: {combined.shape}")
    print("Columns:", list(combined.columns))
    print(combined.head(5).to_string(index=False))

if __name__ == "__main__":
    extract_features()
