import argparse
import datetime as dt
import json

import joblib
import pandas as pd
import yaml

from noctiband.evaluation.metrics import evaluate
from noctiband.models.data import (
    KEYS,
    ROOT,
    check_split,
    feature_columns,
    get_subset,
    load_split,
    load_table,
)
from noctiband.models.registry import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = cfg["name"]

    df = load_table()
    split = load_split()
    check_split(split)
    feats = feature_columns(df)

    train = get_subset(df, split, "train")
    val = get_subset(df, split, "val")  # the test set is NOT touched here
    print(f"{name}: {len(feats)} features, {len(train)} train rows, {len(val)} val rows")

    model = build_model(cfg["model"], cfg.get("params"))
    model.fit(train[feats], train["y_binary"])

    classes = list(model.classes_)
    proba = model.predict_proba(val[feats])
    p_sleep = proba[:, classes.index(1)]
    p_wake = proba[:, classes.index(0)]
    pred = model.predict(val[feats])

    res = evaluate(val["y_binary"], pred)

    # save model + exact feature list (Contract 4)
    models_dir = ROOT / "artifacts/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{name}.joblib")
    with open(models_dir / f"{name}_features.json", "w") as f:
        json.dump({"features": feats}, f, indent=2)

    # save validation predictions (Contract 3)
    pred_dir = ROOT / "artifacts/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out = val[KEYS].copy()
    out["y_true"] = val["y_binary"].to_numpy()
    out["y_pred"] = pred
    out["p_wake"] = p_wake
    out["p_sleep"] = p_sleep
    out.to_csv(pred_dir / f"{name}_val.csv", index=False)

    # append a row to the results table
    res_dir = ROOT / "artifacts/results"
    res_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": name,
        "macro_f1": res["macro_f1"],
        "balanced_accuracy": res["balanced_accuracy"],
        "kappa": res["kappa"],
        "accuracy": res["accuracy"],
        "wake_f1": res["per_class"]["Wake"]["f1"],
        "wake_recall": res["per_class"]["Wake"]["recall"],
        "sleep_f1": res["per_class"]["Sleep"]["f1"],
    }
    table_path = res_dir / "results_val.csv"
    new = pd.DataFrame([row])
    if table_path.exists():
        new = pd.concat([pd.read_csv(table_path), new], ignore_index=True)
    new.to_csv(table_path, index=False)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
