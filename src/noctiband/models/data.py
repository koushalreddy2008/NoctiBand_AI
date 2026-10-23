import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURE_PREFIXES = ("hr_", "acc_", "ctx_", "q_")
KEYS = ["subject_id", "night_id", "epoch_idx"]


def load_table(path=None):
    path = Path(path) if path else ROOT / "artifacts/features/features_v1.csv"
    return pd.read_csv(path)


def feature_columns(df):
    return [
        c
        for c in df.columns
        if c.startswith(FEATURE_PREFIXES)
        and c != "q_flag"
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_split(path=None):
    path = Path(path) if path else ROOT / "artifacts/splits/split_v1.json"
    with open(path) as f:
        return json.load(f)


def check_split(split):
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert not (train & val), "train and val share subjects"
    assert not (train & test), "train and test share subjects"
    assert not (val & test), "val and test share subjects"


def get_subset(df, split, name):
    sub = df[df["subject_id"].isin(split[name])]
    return sub.sort_values(KEYS).reset_index(drop=True)
