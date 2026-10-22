"""
Model registry for NoctiBand AI.
"""

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError:
    DummyClassifier = None
    RandomForestClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None
    SVC = None

SEED = 42


def build_model(kind, params=None):
    params = dict(params or {})
    if kind == "dummy":
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=SEED, **params)
    elif kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=SEED, **params)
    elif kind == "svm":
        from sklearn.svm import SVC
        clf = SVC(probability=True, random_state=SEED, **params)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Imputer and scaler are INSIDE the pipeline, so they are fitted on train data only.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])
