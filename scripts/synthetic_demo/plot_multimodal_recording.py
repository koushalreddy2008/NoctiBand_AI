# SYNTHETIC DEMO ONLY. Signals are generated from ground-truth sleep labels.
# They leak the target. NEVER use them for training or evaluation.
"""
Script: plot_multimodal_recording.py
Purpose: Generate comprehensive overnight multimodal visualization showing:
         1. Heart Rate (BPM)
         2. Blood Pressure (Systolic, Diastolic, MAP in mmHg) with Nocturnal Dipping
         3. SpO2 Oxygen Saturation (%) with Desaturation Event tracking
         4. Expert Sleep Stages (Hypnogram: Wake, N1, N2, N3, REM)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

def plot_multimodal():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    results_dir = root / "results" / "synthetic_demo"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    labels_mat = loadmat(raw_dir / "labels.mat")
    expert_labels = labels_mat["expert_label"].ravel()
    rec_start_text = str(labels_mat["recStart"][0])
    rec_start = pd.Timestamp(rec_start_text).tz_localize("America/New_York")
    rec_start_unix = rec_start.timestamp()

    hr = pd.read_csv(raw_dir / "hr.csv", header=None, names=["timestamp", "heart_rate"])
    bp = pd.read_csv(raw_dir / "blood_pressure.csv")
    spo2 = pd.read_csv(raw_dir / "spo2.csv")

    # Time conversion to UTC datetime
    hr["time"] = pd.to_datetime(hr["timestamp"], unit="s", utc=True)
    bp["time"] = pd.to_datetime(bp["timestamp"], unit="s", utc=True)
    spo2["time"] = pd.to_datetime(spo2["timestamp"], unit="s", utc=True)

    # Epoch timestamps for hypnogram
    epoch_times = [rec_start_unix + i * 30.0 for i in range(len(expert_labels))]
    epoch_datetimes = pd.to_datetime(epoch_times, unit="s", utc=True)

    # -------------------------------------------------------------
    # Plot 1: SpO2 Overnight Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    plt.plot(spo2["time"], spo2["spo2"], color="#0284c7", linewidth=0.9, label="SpO2 (%)")
    plt.axhline(95, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.8, label="Mild Hypoxia Threshold (95%)")
    plt.axhline(90, color="#ef4444", linestyle="--", linewidth=1, alpha=0.8, label="Severe Desaturation Threshold (90%)")
    plt.ylim(86, 101)
    plt.xlabel("Time (UTC)", fontsize=11)
    plt.ylabel("SpO2 (%)", fontsize=11)
    plt.title("Overnight Continuous Pulse Oximetry (SpO2)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    spo2_plot_path = results_dir / "spo2.png"
    plt.savefig(spo2_plot_path, dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Blood Pressure Overnight Dipping Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(14, 5.5))
    plt.plot(bp["time"], bp["systolic"], color="#e11d48", linewidth=1.0, label="Systolic (SBP)", alpha=0.9)
    plt.plot(bp["time"], bp["map"], color="#8b5cf6", linewidth=0.9, linestyle="-.", label="Mean Arterial (MAP)", alpha=0.8)
    plt.plot(bp["time"], bp["diastolic"], color="#2563eb", linewidth=1.0, label="Diastolic (DBP)", alpha=0.9)
    plt.fill_between(bp["time"], bp["diastolic"], bp["systolic"], color="#fbcfe8", alpha=0.25, label="Pulse Pressure")
    plt.axhline(120, color="#dc2626", linestyle=":", linewidth=0.8, alpha=0.6, label="Normal Awake SBP (120 mmHg)")
    plt.xlabel("Time (UTC)", fontsize=11)
    plt.ylabel("Blood Pressure (mmHg)", fontsize=11)
    plt.title("Overnight Blood Pressure Profile & Nocturnal Dipping", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    bp_plot_path = results_dir / "blood_pressure.png"
    plt.savefig(bp_plot_path, dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: 4-Panel Multimodal Sleep Overview
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Subplot 1: Hypnogram
    axes[0].step(epoch_datetimes, expert_labels, where="post", color="#4338ca", linewidth=1.2)
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(["Wake", "N1", "N2", "N3", "REM"], fontsize=10)
    axes[0].set_ylabel("Sleep Stage", fontsize=11)
    axes[0].set_title("NoctiBand AI — Multimodal Overnight Monitoring Overview", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Standard hypnogram convention: Wake at top, N3 at bottom

    # Subplot 2: Heart Rate
    axes[1].plot(hr["time"], hr["heart_rate"], color="#059669", linewidth=0.8)
    axes[1].set_ylabel("Heart Rate (BPM)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Blood Pressure
    axes[2].plot(bp["time"], bp["systolic"], color="#e11d48", linewidth=0.9, label="Systolic (SBP)")
    axes[2].plot(bp["time"], bp["diastolic"], color="#2563eb", linewidth=0.9, label="Diastolic (DBP)")
    axes[2].set_ylabel("BP (mmHg)", fontsize=11)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Subplot 4: SpO2
    axes[3].plot(spo2["time"], spo2["spo2"], color="#0284c7", linewidth=0.8, label="SpO2 (%)")
    axes[3].axhline(95, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.8)
    axes[3].set_ylim(87, 101)
    axes[3].set_ylabel("SpO2 (%)", fontsize=11)
    axes[3].set_xlabel("Time (UTC)", fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    overview_path = results_dir / "multimodal_sleep_overview.png"
    plt.savefig(overview_path, dpi=150)
    plt.close()

    print("Multimodal plots saved:")
    print(" -", spo2_plot_path)
    print(" -", bp_plot_path)
    print(" -", overview_path)

if __name__ == "__main__":
    plot_multimodal()
