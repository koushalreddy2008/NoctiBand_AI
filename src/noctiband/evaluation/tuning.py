import argparse

import numpy as np
import pandas as pd

from noctiband.evaluation.metrics import evaluate
from noctiband.models.data import ROOT
from noctiband.postprocess.smoothing import median_smooth  # written by A; check the name

WINDOWS = [1, 3, 5, 7, 9, 11, 15]  # window=1 means "no smoothing"


def smooth_per_night(df, p_sleep, window):
    """Smooth each night separately. df must be sorted by subject, night, epoch."""
    p = np.asarray(p_sleep, dtype=float)
    out = np.empty(len(p))
    for _, idx in df.groupby(["subject_id", "night_id"], sort=False).indices.items():
        out[idx] = median_smooth(p[idx], window)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    path = ROOT / f"artifacts/predictions/{args.model}_val.csv"
    df = pd.read_csv(path).sort_values(["subject_id", "night_id", "epoch_idx"]).reset_index(drop=True)

    rows = []
    for w in WINDOWS:
        p = smooth_per_night(df, df["p_sleep"], w)
        res = evaluate(df["y_true"], (p >= 0.5).astype(int))
        rows.append({
            "window": w,
            "macro_f1": res["macro_f1"],
            "wake_f1": res["per_class"]["Wake"]["f1"],
            "kappa": res["kappa"],
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    best = table.sort_values(["macro_f1", "window"], ascending=[False, True]).iloc[0]
    print(f"\nBest window on VALIDATION: {int(best['window'])}  (macro-F1 {best['macro_f1']:.4f})")
    (ROOT / "artifacts/results").mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / "artifacts/results/smoothing_val.csv", index=False)


if __name__ == "__main__":
    main()
