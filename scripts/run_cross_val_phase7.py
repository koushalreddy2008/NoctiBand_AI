"""Run leak-free 5-fold cross-validation on 22 train+val subjects using MODEL_FEATURES."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from noctiband.evaluation.harness import evaluate_cv
from noctiband.features.epoch_features import MODEL_FEATURES
from noctiband.io.adapter_v2 import build_dataset_v2

# 4 Test subjects strictly sequestered (never loaded or evaluated here)
TEST_SUBJECTS = {"9106476", "1455390", "5797046", "4426783"}

# Nominal sampling rates for all subjects
SAMPLING_RATES = {
    "5383425": 10.0,
    "8258170": 10.0,
    "9618981": 33.4,
}


def load_train_val_cohort() -> pd.DataFrame:
    """Load curated dataset excluding 4 test-split subjects."""
    df_all, _ = build_dataset_v2("data/raw_v2", min_coverage=0.95, skip_flagged=True)

    # Strictly filter to train + val subjects only
    df_train_val = df_all[~df_all["subject_id"].isin(TEST_SUBJECTS)].copy()
    df_train_val = df_train_val.reset_index(drop=True)

    # Attach sampling_rate metadata
    df_train_val["sampling_rate"] = df_train_val["subject_id"].map(
        lambda s: SAMPLING_RATES.get(str(s), 50.0)
    )
    return df_train_val


def print_eval_result(res, task_name: str) -> None:
    print(f"\n=======================================================")
    print(f"MODEL: {res.estimator_name} | TASK: {task_name}")
    print(f"Subjects: {res.n_subjects} | Epochs: {res.n_epochs} | Wake: {res.n_wake}")
    print("-------------------------------------------------------")
    print("PER-FOLD METRICS:")
    for m in res.per_fold_metrics:
        f_num = m["fold"]
        acc = m.get("accuracy", np.nan)
        mf1 = m["macro_f1"]
        bacc = m["balanced_accuracy"]
        wf1 = m.get("wake_f1", np.nan)
        sf1 = m.get("sleep_f1", np.nan)
        kap = m["cohen_kappa"]
        prauc = m.get("wake_pr_auc", np.nan)
        val_s = ",".join(m["val_subjects"])
        print(
            f"  Fold {f_num} (Val: {val_s}): Acc={acc:.4f}, Macro-F1={mf1:.4f}, "
            f"BalAcc={bacc:.4f}, Wake-F1={wf1}, Sleep-F1={sf1}, Kappa={kap:.4f}, Wake-PRAUC={prauc}"
        )

    print("\nPOOLED METRICS:")
    p = res.pooled_metrics
    print(f"  Accuracy:          {p.get('accuracy', np.nan):.4f}")
    print(f"  Macro-F1:          {p['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {p['balanced_accuracy']:.4f}")
    print(f"  Wake-F1:           {p.get('wake_f1', np.nan)}")
    print(f"  Sleep-F1:          {p.get('sleep_f1', np.nan)}")
    print(f"  Cohen's Kappa:     {p['cohen_kappa']:.4f}")
    print(f"  Wake PR-AUC:       {p.get('wake_pr_auc', np.nan)}")
    print(f"  Confusion Matrix:  {p['confusion_matrix']}")

    print("\nPER-SUBJECT BREAKDOWN:")
    print(res.per_subject_table.to_string(index=False))


def main() -> None:
    df = load_train_val_cohort()
    n_sub = df["subject_id"].nunique()
    print(f"Loaded train+val cohort: {n_sub} subjects, {len(df)} epochs.")
    assert n_sub == 22, f"Expected 22 subjects, got {n_sub}"
    assert not any(s in TEST_SUBJECTS for s in df["subject_id"].unique())

    models = ["majority", "motion_threshold", "logistic_regression", "random_forest"]

    print("\n" + "#" * 60)
    print("PART 1: BINARY CLASSIFICATION (Wake=0 vs Sleep=1)")
    print("#" * 60)
    for model_name in models:
        res = evaluate_cv(
            df=df,
            estimator=model_name,
            feature_columns=MODEL_FEATURES,
            target_col="label_binary",
            n_splits=5,
        )
        print_eval_result(res, "Binary (Wake vs Sleep)")

    print("\n" + "#" * 60)
    print("PART 2: 3-CLASS CLASSIFICATION (Wake=0, NREM=1, REM=2)")
    print("#" * 60)
    for model_name in models:
        res = evaluate_cv(
            df=df,
            estimator=model_name,
            feature_columns=MODEL_FEATURES,
            target_col="label_3class",
            n_splits=5,
        )
        print_eval_result(res, "3-Class (Wake vs NREM vs REM)")


if __name__ == "__main__":
    main()
