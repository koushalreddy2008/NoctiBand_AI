import argparse
import json

import joblib

from noctiband.evaluation.metrics import evaluate
from noctiband.evaluation.tuning import smooth_per_night
from noctiband.models.data import ROOT, check_split, get_subset, load_split, load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / "artifacts/results/test_final.json"
    if out_path.exists() and not args.force:
        raise SystemExit(
            "The test set was already evaluated. Do not tune on it. "
            "Use --force only if you are sure."
        )

    split = load_split()
    check_split(split)
    test = get_subset(load_table(), split, "test")

    model = joblib.load(ROOT / f"artifacts/models/{args.model}.joblib")
    with open(ROOT / f"artifacts/models/{args.model}_features.json") as f:
        feats = json.load(f)["features"]

    classes = list(model.classes_)
    p_sleep = model.predict_proba(test[feats])[:, classes.index(1)]
    p_smooth = smooth_per_night(test, p_sleep, args.window)

    res = evaluate(test["y_binary"], (p_smooth >= 0.5).astype(int))
    res["model"] = args.model
    res["smoothing_window"] = args.window
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
