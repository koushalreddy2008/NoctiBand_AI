"""Phase 9 comprehensive experiments: audits, ablations, nested tuning, and rate tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from noctiband.evaluation.harness import build_pipeline
from noctiband.features.epoch_features import MODEL_FEATURES
from noctiband.models.baselines import MotionThresholdBaseline
from run_cross_val_phase7 import load_train_val_cohort

HR_FEATURES = [f for f in MODEL_FEATURES if f.startswith("hr_")]
MOTION_FEATURES = [f for f in MODEL_FEATURES if f.startswith("motion_")]

TOP_3_MOTION_FEATURES = [
    "motion_enmo_mean_roll_std_w11",
    "motion_mag_mean_roll_std_w11",
    "motion_enmo_mean_roll_std_w5",
]

NON_50HZ_SUBS = ["5383425", "8258170", "9618981"]


def run_correlation_check(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 7: CORRELATION CHECK (|r| > 0.98)")
    print("=" * 70)
    corr_matrix = df[MODEL_FEATURES].corr()
    high_pairs = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.98:
                high_pairs.append((cols[i], cols[j], r))

    print(f"Found {len(high_pairs)} feature pair(s) with |r| > 0.98:")
    for f1, f2, r in sorted(high_pairs, key=lambda x: -abs(x[2])):
        print(f"  {f1} <---> {f2}: r = {r:.4f}")


def run_single_feature_threshold_baselines(df: pd.DataFrame) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("TASK 4: SINGLE-FEATURE THRESHOLD BASELINES (TOP 3 FEATURES)")
    print("=" * 70)
    clean_df = df[df["label_binary"].notna()].reset_index(drop=True)
    y = clean_df["label_binary"].to_numpy(dtype=int)
    groups = clean_df["subject_id"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    results = {}

    for feat in TOP_3_MOTION_FEATURES:
        feat_idx = MODEL_FEATURES.index(feat)
        X_feat = clean_df[[feat]].to_numpy(dtype=float)

        y_true_all, y_pred_all, y_prob_all, sub_all = [], [], [], []
        fold_stats = []

        for fold, (trn, val) in enumerate(gkf.split(X_feat, y, groups), 1):
            pipe = build_pipeline(MotionThresholdBaseline(feature_idx=0))
            pipe.fit(X_feat[trn], y[trn])

            pred = pipe.predict(X_feat[val])
            prob = pipe.predict_proba(X_feat[val])[:, 0]

            est = pipe.named_steps["estimator"]
            fold_stats.append({
                "fold": fold,
                "thresh": est.threshold_,
                "wake_flag_rate": float((pred == 0).mean()),
                "val_bal_acc": float(balanced_accuracy_score(y[val], pred)),
                "val_wake_f1": float(f1_score(y[val], pred, labels=[0], average=None, zero_division=0)[0]),
            })

            y_true_all.extend(y[val])
            y_pred_all.extend(pred)
            y_prob_all.extend(prob)
            sub_all.extend(groups[val])

        yt_arr = np.array(y_true_all)
        yp_arr = np.array(y_pred_all)
        prob_arr = np.array(y_prob_all)
        w_true = (yt_arr == 0).astype(int)

        pooled_roc = float(roc_auc_score(w_true, prob_arr))
        pooled_pr = float(average_precision_score(w_true, prob_arr))
        pooled_wake_f1 = float(f1_score(yt_arr, yp_arr, labels=[0], average=None, zero_division=0)[0])
        pooled_bal_acc = float(balanced_accuracy_score(yt_arr, yp_arr))
        pooled_macro_f1 = float(f1_score(yt_arr, yp_arr, labels=[0, 1], average="macro", zero_division=0))

        # Per-subject ROC-AUC median
        sub_rocs = []
        for s in np.unique(sub_all):
            s_mask = np.array(sub_all) == s
            if len(np.unique(w_true[s_mask])) > 1:
                sub_rocs.append(roc_auc_score(w_true[s_mask], prob_arr[s_mask]))
        med_sub_roc = float(np.median(sub_rocs))

        results[feat] = {
            "pooled_roc_auc": pooled_roc,
            "pooled_pr_auc": pooled_pr,
            "pooled_wake_f1": pooled_wake_f1,
            "pooled_bal_acc": pooled_bal_acc,
            "pooled_macro_f1": pooled_macro_f1,
            "median_sub_roc_auc": med_sub_roc,
            "fold_stats": fold_stats,
            "oof_df": pd.DataFrame({"subject": sub_all, "y_true": yt_arr, "y_pred": yp_arr, "y_prob_wake": prob_arr}),
        }

        print(f"\n--- Feature: {feat} ---")
        print(f"  Pooled ROC-AUC:      {pooled_roc:.4f}")
        print(f"  Pooled PR-AUC:       {pooled_pr:.4f}")
        print(f"  Pooled Wake F1:      {pooled_wake_f1:.4f}")
        print(f"  Pooled Bal Acc:      {pooled_bal_acc:.4f}")
        print(f"  Pooled Macro F1:     {pooled_macro_f1:.4f}")
        print(f"  Median Subj ROC-AUC: {med_sub_roc:.4f}")
        for fs in fold_stats:
            print(f"    Fold {fs['fold']}: thresh={fs['thresh']:.4f}, flag_rate={fs['wake_flag_rate']:.4f}, val_bal_acc={fs['val_bal_acc']:.4f}, val_wake_f1={fs['val_wake_f1']:.4f}")

    return results


def run_feature_ablations(df: pd.DataFrame) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("TASK 5: FEATURE ABLATIONS (MOTION-ONLY, HR-ONLY, ALL 30)")
    print("=" * 70)
    clean_df = df[df["label_binary"].notna()].reset_index(drop=True)
    y = clean_df["label_binary"].to_numpy(dtype=int)
    groups = clean_df["subject_id"].to_numpy()

    feature_subsets = {
        "Motion-only (14)": MOTION_FEATURES,
        "HR-only (16)": HR_FEATURES,
        "Motion+HR (30)": MODEL_FEATURES,
    }

    gkf = GroupKFold(n_splits=5)
    ablation_results = {}

    for subset_name, feats in feature_subsets.items():
        X_sub = clean_df[feats].to_numpy(dtype=float)
        for model_name in ["logistic_regression", "random_forest"]:
            y_true_all, y_pred_all, y_prob_all, sub_all = [], [], [], []

            for fold, (trn, val) in enumerate(gkf.split(X_sub, y, groups), 1):
                pipe = build_pipeline(model_name)
                pipe.fit(X_sub[trn], y[trn])
                pred = pipe.predict(X_sub[val])
                prob = pipe.predict_proba(X_sub[val])[:, 0]

                y_true_all.extend(y[val])
                y_pred_all.extend(pred)
                y_prob_all.extend(prob)
                sub_all.extend(groups[val])

            yt_arr = np.array(y_true_all)
            yp_arr = np.array(y_pred_all)
            prob_arr = np.array(y_prob_all)
            w_true = (yt_arr == 0).astype(int)

            roc = float(roc_auc_score(w_true, prob_arr))
            prauc = float(average_precision_score(w_true, prob_arr))
            wf1 = float(f1_score(yt_arr, yp_arr, labels=[0], average=None, zero_division=0)[0])
            bal = float(balanced_accuracy_score(yt_arr, yp_arr))
            mf1 = float(f1_score(yt_arr, yp_arr, labels=[0, 1], average="macro", zero_division=0))

            sub_rocs = []
            for s in np.unique(sub_all):
                s_mask = np.array(sub_all) == s
                if len(np.unique(w_true[s_mask])) > 1:
                    sub_rocs.append(roc_auc_score(w_true[s_mask], prob_arr[s_mask]))
            med_roc = float(np.median(sub_rocs))

            key = f"{model_name} | {subset_name}"
            ablation_results[key] = {
                "model": model_name,
                "subset": subset_name,
                "roc_auc": roc,
                "pr_auc": prauc,
                "wake_f1": wf1,
                "balanced_acc": bal,
                "macro_f1": mf1,
                "median_sub_roc": med_roc,
                "oof_df": pd.DataFrame({"subject": sub_all, "y_true": yt_arr, "y_pred": yp_arr, "y_prob_wake": prob_arr}),
            }
            print(f"{key:35s}: ROC={roc:.4f}, PR={prauc:.4f}, Wake-F1={wf1:.4f}, BalAcc={bal:.4f}, Macro-F1={mf1:.4f}, MedSubROC={med_roc:.4f}")

    return ablation_results


def run_paired_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    unique_subs = df_a["subject"].unique()
    n_subs = len(unique_subs)

    sub_dict_a = {s: df_a[df_a["subject"] == s] for s in unique_subs}
    sub_dict_b = {s: df_b[df_b["subject"] == s] for s in unique_subs}

    diff_praucs = []
    diff_rocs = []

    for _ in range(n_bootstraps):
        sampled = rng.choice(unique_subs, size=n_subs, replace=True)
        boot_a = pd.concat([sub_dict_a[s] for s in sampled], ignore_index=True)
        boot_b = pd.concat([sub_dict_b[s] for s in sampled], ignore_index=True)

        w_true = (boot_a["y_true"].to_numpy() == 0).astype(int)
        if len(np.unique(w_true)) <= 1:
            continue

        pr_a = average_precision_score(w_true, boot_a["y_prob_wake"].to_numpy())
        pr_b = average_precision_score(w_true, boot_b["y_prob_wake"].to_numpy())
        diff_praucs.append(pr_a - pr_b)

        roc_a = roc_auc_score(w_true, boot_a["y_prob_wake"].to_numpy())
        roc_b = roc_auc_score(w_true, boot_b["y_prob_wake"].to_numpy())
        diff_rocs.append(roc_a - roc_b)

    ci_pr = (np.percentile(diff_praucs, 2.5), np.percentile(diff_praucs, 97.5))
    ci_roc = (np.percentile(diff_rocs, 2.5), np.percentile(diff_rocs, 97.5))

    print(f"\nPaired Bootstrap Difference: [{label_a}] MINUS [{label_b}]")
    print(f"  Mean Delta PR-AUC:  {np.mean(diff_praucs):+.4f} | 95% CI: [{ci_pr[0]:+.4f}, {ci_pr[1]:+.4f}]")
    print(f"  Mean Delta ROC-AUC: {np.mean(diff_rocs):+.4f} | 95% CI: [{ci_roc[0]:+.4f}, {ci_roc[1]:+.4f}]")


def run_nested_threshold_tuning(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 8: NESTED THRESHOLD TUNING (INNER GROUPKFOLD ON TRAINING FOLDS ONLY)")
    print("=" * 70)
    clean_df = df[df["label_binary"].notna()].reset_index(drop=True)
    X = clean_df[MODEL_FEATURES].to_numpy(dtype=float)
    y = clean_df["label_binary"].to_numpy(dtype=int)
    groups = clean_df["subject_id"].to_numpy()

    outer_gkf = GroupKFold(n_splits=5)
    candidates = np.linspace(0.10, 0.90, 81)

    for model_name in ["logistic_regression", "random_forest"]:
        print(f"\n--- MODEL: {model_name} ---")
        y_true_all = []
        y_pred_fixed_all = []
        y_pred_tuned_all = []
        chosen_thresholds = []
        fold_results = []

        for fold, (outer_trn, outer_val) in enumerate(outer_gkf.split(X, y, groups), 1):
            X_trn, y_trn, g_trn = X[outer_trn], y[outer_trn], groups[outer_trn]
            X_val, y_val = X[outer_val], y[outer_val]

            # Inner GroupKFold to find optimal threshold maximizing Wake F1
            inner_gkf = GroupKFold(n_splits=4)
            inner_y_true = []
            inner_y_prob = []

            for in_trn, in_val in inner_gkf.split(X_trn, y_trn, g_trn):
                in_pipe = build_pipeline(model_name)
                in_pipe.fit(X_trn[in_trn], y_trn[in_trn])
                in_prob = in_pipe.predict_proba(X_trn[in_val])[:, 0]
                inner_y_true.extend(y_trn[in_val])
                inner_y_prob.extend(in_prob)

            in_yt = np.array(inner_y_true)
            in_yp = np.array(inner_y_prob)

            best_thresh = 0.5
            best_wake_f1 = -1.0
            for th in candidates:
                # Wake is class 0: pred = 0 if prob_wake >= th else 1
                pred_binary = np.where(in_yp >= th, 0, 1)
                f1_w = f1_score(in_yt, pred_binary, labels=[0], average=None, zero_division=0)[0]
                if f1_w > best_wake_f1:
                    best_wake_f1 = f1_w
                    best_thresh = float(th)

            chosen_thresholds.append(best_thresh)

            # Fit on full outer training fold
            outer_pipe = build_pipeline(model_name)
            outer_pipe.fit(X_trn, y_trn)

            val_prob_wake = outer_pipe.predict_proba(X_val)[:, 0]
            pred_fixed = np.where(val_prob_wake >= 0.5, 0, 1)
            pred_tuned = np.where(val_prob_wake >= best_thresh, 0, 1)

            f1_fix = f1_score(y_val, pred_fixed, labels=[0], average=None, zero_division=0)[0]
            f1_tune = f1_score(y_val, pred_tuned, labels=[0], average=None, zero_division=0)[0]
            bal_fix = balanced_accuracy_score(y_val, pred_fixed)
            bal_tune = balanced_accuracy_score(y_val, pred_tuned)

            fold_results.append({
                "fold": fold,
                "chosen_thresh": best_thresh,
                "fixed_wake_f1": f1_fix,
                "tuned_wake_f1": f1_tune,
                "fixed_bal_acc": bal_fix,
                "tuned_bal_acc": bal_tune,
            })

            y_true_all.extend(y_val)
            y_pred_fixed_all.extend(pred_fixed)
            y_pred_tuned_all.extend(pred_tuned)

        print(pd.DataFrame(fold_results).to_string(index=False))

        yt_arr = np.array(y_true_all)
        p_fix_arr = np.array(y_pred_fixed_all)
        p_tune_arr = np.array(y_pred_tuned_all)

        print("\nPOOLED OUTER-FOLD COMPARISON:")
        print(f"  Fixed 0.5: Wake-F1 = {f1_score(yt_arr, p_fix_arr, labels=[0], average=None, zero_division=0)[0]:.4f}, BalAcc = {balanced_accuracy_score(yt_arr, p_fix_arr):.4f}, Macro-F1 = {f1_score(yt_arr, p_fix_arr, average='macro', zero_division=0):.4f}")
        print(f"  Tuned Th:  Wake-F1 = {f1_score(yt_arr, p_tune_arr, labels=[0], average=None, zero_division=0)[0]:.4f}, BalAcc = {balanced_accuracy_score(yt_arr, p_tune_arr):.4f}, Macro-F1 = {f1_score(yt_arr, p_tune_arr, average='macro', zero_division=0):.4f}")
        print(f"  Chosen thresholds per fold: {chosen_thresholds}")


def run_rate_robustness_task9(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TASK 9: RATE ROBUSTNESS FOR NON-50Hz SUBJECTS (5383425, 8258170, 9618981)")
    print("=" * 70)
    clean_df = df[df["label_binary"].notna()].reset_index(drop=True)

    for sub in NON_50HZ_SUBS:
        fs = clean_df.loc[clean_df["subject_id"] == sub, "sampling_rate"].iloc[0]
        sub_mask = clean_df["subject_id"] == sub
        X_sub = clean_df.loc[sub_mask, MODEL_FEATURES].to_numpy(dtype=float)
        y_sub = (clean_df.loc[sub_mask, "label_binary"].to_numpy(dtype=int) == 0).astype(int)

        # Cohort (a): All other 21 subjects
        trn_a = clean_df["subject_id"] != sub
        X_trn_a = clean_df.loc[trn_a, MODEL_FEATURES].to_numpy(dtype=float)
        y_trn_a = clean_df.loc[trn_a, "label_binary"].to_numpy(dtype=int)

        # Cohort (b): The 19 50 Hz subjects only
        trn_b = (clean_df["subject_id"] != sub) & (~clean_df["subject_id"].isin(NON_50HZ_SUBS))
        X_trn_b = clean_df.loc[trn_b, MODEL_FEATURES].to_numpy(dtype=float)
        y_trn_b = clean_df.loc[trn_b, "label_binary"].to_numpy(dtype=int)

        print(f"\n--- Subject {sub} ({fs} Hz, N={len(y_sub)}, Wake={y_sub.sum()}) ---")
        for model_name in ["logistic_regression", "random_forest"]:
            pipe_a = build_pipeline(model_name).fit(X_trn_a, y_trn_a)
            prob_a = pipe_a.predict_proba(X_sub)[:, 0]
            roc_a = roc_auc_score(y_sub, prob_a)
            pr_a = average_precision_score(y_sub, prob_a)

            pipe_b = build_pipeline(model_name).fit(X_trn_b, y_trn_b)
            prob_b = pipe_b.predict_proba(X_sub)[:, 0]
            roc_b = roc_auc_score(y_sub, prob_b)
            pr_b = average_precision_score(y_sub, prob_b)

            print(f"  {model_name:20s}:")
            print(f"    (a) Train on all 21: ROC-AUC = {roc_a:.4f}, PR-AUC = {pr_a:.4f}")
            print(f"    (b) Train on 19 50Hz: ROC-AUC = {roc_b:.4f}, PR-AUC = {pr_b:.4f}")
            diff_roc = roc_b - roc_a
            diff_pr = pr_b - pr_a
            status = "no evidence of" if abs(diff_roc) < 0.05 and abs(diff_pr) < 0.05 else "evidence of"
            print(f"    Delta (b - a): ROC = {diff_roc:+.4f}, PR = {diff_pr:+.4f} ({status} rate bias)")


def main() -> None:
    df = load_train_val_cohort()
    run_correlation_check(df)
    single_res = run_single_feature_threshold_baselines(df)
    ablation_res = run_feature_ablations(df)

    best_single_feat = TOP_3_MOTION_FEATURES[0]
    df_best_single = single_res[best_single_feat]["oof_df"]
    df_lr_30 = ablation_res["logistic_regression | Motion+HR (30)"]["oof_df"]
    df_lr_motion = ablation_res["logistic_regression | Motion-only (14)"]["oof_df"]

    print("\n" + "=" * 70)
    print("TASK 6: PAIRED SUBJECT-LEVEL BOOTSTRAP (95% CI)")
    print("=" * 70)
    run_paired_bootstrap(df_lr_30, df_best_single, "LR(30)", f"Best Single ({best_single_feat})")
    run_paired_bootstrap(df_lr_30, df_lr_motion, "LR(30)", "LR(Motion-only)")

    run_nested_threshold_tuning(df)
    run_rate_robustness_task9(df)


if __name__ == "__main__":
    main()
