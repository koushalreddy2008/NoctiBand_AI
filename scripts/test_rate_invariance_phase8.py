"""Evaluate sampling rate invariance on real data (downsampling and cross-frequency testing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from noctiband.evaluation.harness import build_pipeline
from noctiband.features.epoch_features import MODEL_FEATURES, build_epoch_table
from noctiband.io.adapter_v2 import load_subject_v2
from run_cross_val_phase7 import load_train_val_cohort

NON_50HZ_SUBJECTS = ["5383425", "8258170", "9618981"]
DOWNSAMPLE_50HZ_SUBJECTS = ["1360686", "1818471", "2638030", "3997827", "4314139"]


def run_downsampling_test() -> None:
    print("=" * 65)
    print("TASK 5: DOWNSAMPLING RAW MOTION (50 Hz -> 10 Hz) FOR 5 SUBJECTS")
    print("=" * 65)

    df_full = load_train_val_cohort()
    # Train model on all 22 subjects or standard pipeline
    pipe_lr = build_pipeline("logistic_regression")
    pipe_rf = build_pipeline("random_forest")

    clean_mask = df_full["label_binary"].notna()
    X_train = df_full.loc[clean_mask, MODEL_FEATURES].to_numpy(dtype=float)
    y_train = df_full.loc[clean_mask, "label_binary"].to_numpy(dtype=int)

    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)

    feature_diffs: dict[str, list[float]] = {f: [] for f in MODEL_FEATURES}
    lr_prob_diffs: list[float] = []
    rf_prob_diffs: list[float] = []

    for sub in DOWNSAMPLE_50HZ_SUBJECTS:
        night_orig = load_subject_v2("data/raw_v2", sub, allow_partial=True)
        tbl_orig = build_epoch_table(night_orig)

        # Downsample motion raw 50 Hz to 10 Hz (every 5th sample)
        mot_down = night_orig.motion.iloc[::5].reset_index(drop=True)
        night_down = load_subject_v2("data/raw_v2", sub, allow_partial=True)
        object.__setattr__(night_down, "motion", mot_down)
        tbl_down = build_epoch_table(night_down)

        X_orig = tbl_orig[MODEL_FEATURES].to_numpy(dtype=float)
        X_down = tbl_down[MODEL_FEATURES].to_numpy(dtype=float)

        # Compute MAD per feature
        for j, feat in enumerate(MODEL_FEATURES):
            v_orig = X_orig[:, j]
            v_down = X_down[:, j]
            valid = np.isfinite(v_orig) & np.isfinite(v_down)
            if valid.any():
                mad = float(np.mean(np.abs(v_orig[valid] - v_down[valid])))
                feature_diffs[feat].append(mad)

        # Predict Wake probability (class 0)
        p_lr_orig = pipe_lr.predict_proba(X_orig)[:, 0]
        p_lr_down = pipe_lr.predict_proba(X_down)[:, 0]
        lr_prob_diffs.extend(np.abs(p_lr_orig - p_lr_down).tolist())

        p_rf_orig = pipe_rf.predict_proba(X_orig)[:, 0]
        p_rf_down = pipe_rf.predict_proba(X_down)[:, 0]
        rf_prob_diffs.extend(np.abs(p_rf_orig - p_rf_down).tolist())

    print("\nMEAN ABSOLUTE DIFFERENCE (50 Hz vs 10 Hz) PER FEATURE:")
    mad_summary = []
    for feat in MODEL_FEATURES:
        mean_mad = float(np.mean(feature_diffs[feat])) if feature_diffs[feat] else np.nan
        mad_summary.append({"feature": feat, "mean_absolute_difference": round(mean_mad, 5)})
    mad_df = pd.DataFrame(mad_summary).sort_values("mean_absolute_difference", ascending=False)
    print(mad_df.to_string(index=False))

    mean_lr_diff = float(np.mean(lr_prob_diffs))
    mean_rf_diff = float(np.mean(rf_prob_diffs))
    print(f"\nPREDICTED WAKE PROBABILITY CHANGE (MAD across {len(lr_prob_diffs)} epochs):")
    print(f"  Logistic Regression: {mean_lr_diff:.4f} ({mean_lr_diff*100:.2f}% probability shift)")
    print(f"  Random Forest:       {mean_rf_diff:.4f} ({mean_rf_diff*100:.2f}% probability shift)")


def run_cross_frequency_train_test() -> None:
    print("\n" + "=" * 65)
    print("TASK 6: TRAIN ON 19 (50 Hz) SUBJECTS ONLY -> TEST ON 10/33 Hz")
    print("=" * 65)

    df_full = load_train_val_cohort()
    clean_mask = df_full["label_binary"].notna()
    df_clean = df_full[clean_mask].copy()

    train_mask = ~df_clean["subject_id"].isin(NON_50HZ_SUBJECTS)
    test_mask = df_clean["subject_id"].isin(NON_50HZ_SUBJECTS)

    X_train = df_clean.loc[train_mask, MODEL_FEATURES].to_numpy(dtype=float)
    y_train = df_clean.loc[train_mask, "label_binary"].to_numpy(dtype=int)

    for model_name in ["majority", "motion_threshold", "logistic_regression", "random_forest"]:
        pipe = build_pipeline(model_name)
        pipe.fit(X_train, y_train)

        print(f"\n--- MODEL: {model_name} (Trained on 19 x 50 Hz subjects) ---")
        for sub in NON_50HZ_SUBJECTS:
            sub_mask = df_clean["subject_id"] == sub
            X_sub = df_clean.loc[sub_mask, MODEL_FEATURES].to_numpy(dtype=float)
            y_sub = df_clean.loc[sub_mask, "label_binary"].to_numpy(dtype=int)

            y_pred = pipe.predict(X_sub)
            prob_wake = pipe.predict_proba(X_sub)[:, 0] if hasattr(pipe, "predict_proba") else None

            acc = accuracy_score(y_sub, y_pred)
            bal_acc = balanced_accuracy_score(y_sub, y_pred)
            mf1 = f1_score(y_sub, y_pred, average="macro", zero_division=0)
            wf1 = f1_score(y_sub, y_pred, labels=[0], average=None, zero_division=0)[0]
            prauc = average_precision_score((y_sub == 0).astype(int), prob_wake) if prob_wake is not None else np.nan

            fs = 10.0 if sub in ["5383425", "8258170"] else 33.4
            print(
                f"Subject {sub} ({fs} Hz, N={len(y_sub)}, Wake={(y_sub==0).sum()}): "
                f"Acc={acc:.4f}, BalAcc={bal_acc:.4f}, Macro-F1={mf1:.4f}, Wake-F1={wf1:.4f}, Wake-PRAUC={prauc:.4f}"
            )


if __name__ == "__main__":
    run_downsampling_test()
    run_cross_frequency_train_test()
