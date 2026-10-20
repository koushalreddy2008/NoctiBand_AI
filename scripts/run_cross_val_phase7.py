"""Run leak-free 5-fold cross-validation on 22 train+val subjects with full Phase 8 reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from noctiband.evaluation.harness import evaluate_cv
from noctiband.features.epoch_features import MODEL_FEATURES
from noctiband.io.adapter_v2 import build_dataset_v2

TEST_SUBJECTS = {"9106476", "1455390", "5797046", "4426783"}

SAMPLING_RATES = {
    "5383425": 10.0,
    "8258170": 10.0,
    "9618981": 33.4,
}


def load_train_val_cohort() -> pd.DataFrame:
    """Load curated dataset excluding 4 test-split subjects."""
    df_all, _ = build_dataset_v2("data/raw_v2", min_coverage=0.95, skip_flagged=True)
    df_train_val = df_all[~df_all["subject_id"].isin(TEST_SUBJECTS)].copy()
    df_train_val = df_train_val.reset_index(drop=True)
    df_train_val["sampling_rate"] = df_train_val["subject_id"].map(
        lambda s: SAMPLING_RATES.get(str(s), 50.0)
    )
    return df_train_val


def compute_subject_bootstrap_ci(
    res_df: pd.DataFrame,
    classes: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute subject-level bootstrap 95% confidence intervals."""
    rng = np.random.default_rng(seed)
    unique_subs = res_df["subject"].unique()
    n_subs = len(unique_subs)

    sub_groups = {s: res_df[res_df["subject"] == s] for s in unique_subs}

    boot_macro_f1s = []
    boot_wake_praucs = []

    for _ in range(n_bootstraps):
        sampled_subs = rng.choice(unique_subs, size=n_subs, replace=True)
        sampled_dfs = [sub_groups[s] for s in sampled_subs]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)

        y_t = boot_df["y_true"].to_numpy()
        y_p = boot_df["y_pred"].to_numpy()

        mf1 = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
        boot_macro_f1s.append(mf1)

        if "y_prob_wake" in boot_df.columns:
            w_true = (y_t == 0).astype(int)
            w_prob = boot_df["y_prob_wake"].to_numpy()
            if len(np.unique(w_true)) > 1:
                prauc = float(average_precision_score(w_true, w_prob))
                boot_wake_praucs.append(prauc)

    ci_macro_f1 = (
        round(float(np.percentile(boot_macro_f1s, 2.5)), 4),
        round(float(np.percentile(boot_macro_f1s, 97.5)), 4),
    )
    ci_wake_prauc = (
        round(float(np.percentile(boot_wake_praucs, 2.5)), 4),
        round(float(np.percentile(boot_wake_praucs, 97.5)), 4),
    ) if boot_wake_praucs else (np.nan, np.nan)

    return {"macro_f1_ci": ci_macro_f1, "wake_pr_auc_ci": ci_wake_prauc}


def print_full_eval_report(res, task_name: str, df_clean: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"MODEL: {res.estimator_name} | TASK: {task_name}")
    print(f"Subjects: {res.n_subjects} | Epochs: {res.n_epochs} | Wake: {res.n_wake}")
    print(f"{'-'*70}")

    print("PER-FOLD TABLE:")
    fold_rows = []
    for m in res.per_fold_metrics:
        fold_rows.append({
            "fold": m["fold"],
            "accuracy": m.get("accuracy", np.nan),
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "wake_f1": m.get("wake_f1", np.nan),
            "wake_roc_auc": m.get("wake_roc_auc", np.nan),
            "wake_pr_auc": m.get("wake_pr_auc", np.nan),
            "val_subjects": ",".join(str(s) for s in m["val_subjects"]),
        })
    print(pd.DataFrame(fold_rows).to_string(index=False))

    print("\nPOOLED METRICS (All Out-Of-Fold Predictions Combined):")
    p = res.pooled_metrics
    print(f"  Accuracy:          {p.get('accuracy', np.nan):.4f}")
    print(f"  Macro-F1:          {p['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {p['balanced_accuracy']:.4f}")
    print(f"  Wake-F1:           {p.get('wake_f1', np.nan)}")
    print(f"  Sleep-F1:          {p.get('sleep_f1', np.nan)}")
    print(f"  Cohen's Kappa:     {p['cohen_kappa']:.4f}")
    print(f"  Wake ROC-AUC:      {p.get('wake_roc_auc', np.nan)}")
    print(f"  Wake PR-AUC:       {p.get('wake_pr_auc', np.nan)}")
    print(f"  Confusion Matrix:  {p['confusion_matrix']}")

    # Subject-level bootstrap CIs
    # Build out-of-fold dataframe for bootstrap
    classes = np.array([0, 1]) if "Binary" in task_name else np.array([0, 1, 2])
    # Extract prediction records
    res_sub_df = res.per_subject_table
    print("\nPER-SUBJECT TABLE (rate, n_wake, n_epochs, bal acc, Wake F1, Wake PR-AUC):")
    cols_to_print = ["subject", "sampling_rate", "n_wake", "n_epochs", "balanced_accuracy", "wake_f1", "wake_pr_auc"]
    available_cols = [c for c in cols_to_print if c in res_sub_df.columns]
    print(res_sub_df[available_cols].to_string(index=False))

    # Subjects with balanced accuracy < 0.55
    low_bal = res_sub_df[res_sub_df["balanced_accuracy"] < 0.55]
    print(f"\nSubjects with Balanced Accuracy < 0.55 (Total: {len(low_bal)}):")
    if not low_bal.empty:
        print(low_bal[["subject", "sampling_rate", "n_wake", "n_epochs", "balanced_accuracy"]].to_string(index=False))
    else:
        print("  None.")


def main() -> None:
    df = load_train_val_cohort()
    models = ["majority", "motion_threshold", "logistic_regression", "random_forest"]

    print("\n" + "#" * 70)
    print("PHASE 8 CROSS-VALIDATION: BINARY CLASSIFICATION (Wake=0 vs Sleep=1)")
    print("#" * 70)
    for model_name in models:
        res = evaluate_cv(
            df=df,
            estimator=model_name,
            feature_columns=MODEL_FEATURES,
            target_col="label_binary",
            n_splits=5,
        )
        print_full_eval_report(res, "Binary (Wake vs Sleep)", df)

        # Compute Bootstrap CIs for LR and RF
        if model_name in ["logistic_regression", "random_forest"]:
            # Reconstruct prediction table for bootstrap
            # Re-run or extract
            pass

    print("\n" + "#" * 70)
    print("PHASE 8 CROSS-VALIDATION: 3-CLASS CLASSIFICATION (Wake=0, NREM=1, REM=2)")
    print("#" * 70)
    for model_name in models:
        res = evaluate_cv(
            df=df,
            estimator=model_name,
            feature_columns=MODEL_FEATURES,
            target_col="label_3class",
            n_splits=5,
        )
        print_full_eval_report(res, "3-Class (Wake vs NREM vs REM)", df)


if __name__ == "__main__":
    main()
