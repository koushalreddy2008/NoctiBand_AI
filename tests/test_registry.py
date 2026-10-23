import numpy as np
import pandas as pd
import pytest

from noctiband.models.registry import build_model


def test_all_models_fit_and_predict_with_missing_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=["hr_a", "hr_b", "acc_a", "q_a"])
    X.iloc[::17, 0] = np.nan  # simulate missing heart rate
    y = (X["acc_a"] > 0).astype(int)
    for kind in ["dummy", "logreg", "random_forest", "svm"]:
        try:
            model = build_model(kind)
            model.fit(X, y)
            assert model.predict(X).shape == (200,)
        except ImportError as e:
            if "Application Control policy" in str(e):
                pytest.skip(
                    f"Skipping {kind} due to Windows Smart App Control blocking native extension"
                )
            raise
