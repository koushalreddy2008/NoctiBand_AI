# SYNTHETIC DEMO ONLY. Signals are generated from ground-truth sleep labels.
# They leak the target. NEVER use them for training or evaluation.
"""
Script: generate_overnight_biometrics.py
Purpose: Generate realistic, physiologically grounded overnight Blood Pressure (BP)
         and SpO2 pulse oximetry data synchronized with PhysioNet BIDSleep (Bidslab00 / night 3).

Physiological Modeling:
- Night duration: 993 30-second epochs (~8.28 hours) matching data/raw/labels.mat and data/raw/hr.csv.
- Nocturnal Blood Pressure Dipping:
    * Wake (Stage 0): SBP ~118-124 mmHg, DBP ~74-80 mmHg (high sympathetic tone, awake baseline).
    * N1 (Stage 1): SBP ~112-117 mmHg, DBP ~70-75 mmHg.
    * N2 (Stage 2): SBP ~105-112 mmHg, DBP ~65-72 mmHg (stable moderate dipping).
    * N3 (Stage 3): SBP ~98-105 mmHg, DBP ~60-66 mmHg (maximal parasympathetic dominance, 15-20% dipping).
    * REM (Stage 4): SBP ~112-124 mmHg, DBP ~72-79 mmHg (autonomic instability, phasic sympathetic bursts).
- Heart Rate Coupling: Baroreflex and autonomic coupling with data/raw/hr.csv.
- SpO2 (Pulse Oximetry):
    * Baseline healthy sleep: 96.5% - 98.5%.
    * Stable high SpO2 in N3 slow-wave sleep.
    * Occasional mild hypopnea/apnea desaturations (drops to 91-93% lasting 15-30s with rapid resaturation)
      allowing calculation of Oxygen Desaturation Index (ODI).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

def generate_biometrics():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    results_dir = root / "results" / "synthetic_demo"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ground-truth labels and heart rate
    labels_mat = loadmat(raw_dir / "labels.mat")
    expert_labels = labels_mat["expert_label"].ravel()
    rec_start_text = str(labels_mat["recStart"][0])
    rec_start = pd.Timestamp(rec_start_text).tz_localize("America/New_York")
    rec_start_unix = rec_start.timestamp()
    
    total_epochs = len(expert_labels)
    total_duration_sec = total_epochs * 30
    rec_end_unix = rec_start_unix + total_duration_sec

    # Load HR
    hr_df = pd.read_csv(raw_dir / "hr.csv", header=None, names=["timestamp", "heart_rate"])
    
    print(f"Loaded {total_epochs} sleep stages. Rec start: {rec_start_unix}, Rec end: {rec_end_unix}")
    print(f"Loaded {len(hr_df)} HR points. HR range: [{hr_df['heart_rate'].min()}, {hr_df['heart_rate'].max()}]")

    # Set random seed for reproducibility
    np.random.seed(42)

    # -------------------------------------------------------------
    # 2. GENERATE BLOOD PRESSURE (Continuous Optical / Cuffless BP)
    # Timestamps matching data/raw/hr.csv
    # -------------------------------------------------------------
    bp_records = []
    
    # Calculate baseline HR for reference
    mean_hr = hr_df["heart_rate"].mean()

    # Pre-generate low-frequency vasomotor Mayer wave oscillations (0.05 - 0.1 Hz)
    time_points = hr_df["timestamp"].values
    dt_from_start = time_points - rec_start_unix
    mayer_wave = 2.5 * np.sin(2 * np.pi * dt_from_start / 20.0) + 1.5 * np.cos(2 * np.pi * dt_from_start / 60.0)

    for idx, row in hr_df.iterrows():
        t = row["timestamp"]
        hr = row["heart_rate"]
        
        # Determine corresponding epoch and sleep stage
        epoch_idx = int(np.floor((t - rec_start_unix) / 30.0))
        if 0 <= epoch_idx < total_epochs:
            stage = expert_labels[epoch_idx]
        else:
            stage = 0  # pre-sleep / wake
            
        # Target BP according to circadian sleep stage
        if stage == 0:    # Wake
            base_sbp = 120.0
            base_dbp = 78.0
            stage_noise = np.random.normal(0, 2.5)
        elif stage == 1:  # N1
            base_sbp = 114.0
            base_dbp = 73.0
            stage_noise = np.random.normal(0, 1.8)
        elif stage == 2:  # N2 (Stable light sleep, ~10% dip)
            base_sbp = 108.0
            base_dbp = 68.0
            stage_noise = np.random.normal(0, 1.2)
        elif stage == 3:  # N3 (Deep slow wave, maximal ~18% dip)
            base_sbp = 101.0
            base_dbp = 62.0
            stage_noise = np.random.normal(0, 0.9)
        elif stage == 4:  # REM (Sympathetic variability, phasic surges)
            base_sbp = 116.0
            base_dbp = 75.0
            stage_noise = np.random.normal(0, 3.2)
        else:
            base_sbp = 118.0
            base_dbp = 76.0
            stage_noise = 0.0

        # HR-coupling: baroreceptor / autonomic delta
        hr_delta = hr - mean_hr
        sbp_hr_effect = 0.45 * hr_delta
        dbp_hr_effect = 0.28 * hr_delta

        # Vasomotor oscillations
        mw = mayer_wave[idx]

        # Calculate SBP, DBP, MAP, Pulse Pressure
        sbp = base_sbp + sbp_hr_effect + 0.6 * mw + stage_noise
        dbp = base_dbp + dbp_hr_effect + 0.3 * mw + 0.7 * stage_noise
        
        # Physiological safety bounds
        sbp = np.clip(sbp, 85.0, 150.0)
        dbp = np.clip(dbp, 50.0, 95.0)
        if dbp >= sbp - 20.0:
            dbp = sbp - 25.0

        pulse_pressure = sbp - dbp
        map_val = dbp + (pulse_pressure / 3.0)

        bp_records.append({
            "timestamp": t,
            "systolic": round(float(sbp), 1),
            "diastolic": round(float(dbp), 1),
            "map": round(float(map_val), 1),
            "pulse_pressure": round(float(pulse_pressure), 1)
        })

    bp_df = pd.DataFrame(bp_records)
    bp_path = raw_dir / "blood_pressure.csv"
    bp_copy_path = raw_dir / "bp.csv"
    bp_df.to_csv(bp_path, index=False)
    bp_df.to_csv(bp_copy_path, index=False)
    print(f"Created {bp_path} and {bp_copy_path} with {len(bp_df)} readings.")

    # -------------------------------------------------------------
    # 3. GENERATE SPO2 (1 Hz Continuous Nocturnal Pulse Oximetry)
    # Covering full recording window: start to end (1 sample/sec)
    # -------------------------------------------------------------
    # Timestamp grid at 1 Hz from first HR timestamp to last HR timestamp
    start_time = int(np.floor(hr_df["timestamp"].min()))
    end_time = int(np.ceil(hr_df["timestamp"].max()))
    sec_timestamps = np.arange(start_time, end_time + 1, 1.0)
    
    # Identify hypopnea desaturation windows (simulate 6-8 nocturnal desaturation events,
    # common in clinical sleep studies, predominantly during REM or stage shifts)
    # Select epochs in REM (stage 4) and N2 (stage 2) for brief desaturations
    desat_events = [
        # (epoch_num, duration_sec, drop_depth_pct)
        (120, 24, 4.5),   # early night N2
        (280, 28, 5.2),   # REM 1
        (460, 32, 6.0),   # REM 2 (moderate drop to ~92%)
        (610, 22, 4.0),   # N2 transition
        (750, 35, 6.8),   # REM 3 (drop to ~91%)
        (890, 26, 4.8),   # late night REM
    ]
    
    # Map desaturations to second-level timestamps
    desat_profiles = np.zeros(len(sec_timestamps))
    for ep, dur, depth in desat_events:
        ep_t_start = rec_start_unix + ep * 30.0 + 5.0
        mask = (sec_timestamps >= ep_t_start) & (sec_timestamps < ep_t_start + dur)
        if np.any(mask):
            event_indices = np.where(mask)[0]
            # U-shaped or V-shaped desaturation curve
            t_norm = np.linspace(0, np.pi, len(event_indices))
            dip = depth * np.sin(t_norm)
            desat_profiles[event_indices] += dip

    # Slow baseline wander
    baseline_wander = 0.4 * np.sin(2 * np.pi * (sec_timestamps - rec_start_unix) / 7200.0)

    spo2_records = []
    for i, t in enumerate(sec_timestamps):
        epoch_idx = int(np.floor((t - rec_start_unix) / 30.0))
        if 0 <= epoch_idx < total_epochs:
            stage = expert_labels[epoch_idx]
        else:
            stage = 0

        # Stage baseline SpO2
        if stage == 0:      # Wake
            base_spo2 = 98.2
            noise = np.random.normal(0, 0.25)
        elif stage == 1:    # N1
            base_spo2 = 97.8
            noise = np.random.normal(0, 0.2)
        elif stage == 2:    # N2
            base_spo2 = 97.6
            noise = np.random.normal(0, 0.15)
        elif stage == 3:    # N3 (deep, very stable respiration)
            base_spo2 = 97.9
            noise = np.random.normal(0, 0.1)
        elif stage == 4:    # REM (higher respiratory irregularity)
            base_spo2 = 97.1
            noise = np.random.normal(0, 0.35)
        else:
            base_spo2 = 98.0
            noise = 0.0

        val = base_spo2 + baseline_wander[i] - desat_profiles[i] + noise
        val = np.clip(val, 88.0, 100.0)

        spo2_records.append({
            "timestamp": t,
            "spo2": round(float(val), 1)
        })

    spo2_df = pd.DataFrame(spo2_records)
    spo2_path = raw_dir / "spo2.csv"
    spo2_df.to_csv(spo2_path, index=False)
    print(f"Created {spo2_path} with {len(spo2_df)} 1-Hz readings.")

    return bp_df, spo2_df

if __name__ == "__main__":
    generate_biometrics()
