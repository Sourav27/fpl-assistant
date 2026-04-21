# tests/test_tune.py
import numpy as np
import pandas as pd
import pytest
from src.pipeline.tune import tune_position_model, validate_training_data

RNG = np.random.default_rng(42)
N = 300

def _make_temporal_df(n=N, pos="MID"):
    seasons = ["2022-23"] * (n // 2) + ["2023-24"] * (n // 2)
    gws = list(range(1, n // 2 + 1)) * 2
    return pd.DataFrame({
        "season": seasons,
        "GW": gws,
        "f0": RNG.random(n), "f1": RNG.random(n),
        "f2": RNG.random(n), "f3": RNG.random(n), "f4": RNG.random(n),
        "total_points": RNG.integers(0, 12, n).astype(float),
    })

DF = _make_temporal_df()
X_TRAIN = DF[["f0", "f1", "f2", "f3", "f4"]]
Y_TRAIN = DF["total_points"]
FEAT_COLS = ["f0", "f1", "f2", "f3", "f4"]


def test_validate_passes_good_data():
    validate_training_data(DF, FEAT_COLS, pos="MID", min_rows=100)


def test_validate_raises_insufficient_rows():
    with pytest.raises(ValueError, match="insufficient rows"):
        validate_training_data(DF.head(50), FEAT_COLS, pos="MID", min_rows=100)


def test_validate_raises_all_zero_feature():
    bad = DF.copy()
    bad["f0"] = 0.0
    with pytest.raises(ValueError, match="all-zero"):
        validate_training_data(bad, FEAT_COLS, pos="MID", min_rows=100)


def test_validate_raises_on_nan_rho_risk():
    const = DF.copy()
    const["total_points"] = 5.0
    validate_training_data(const, FEAT_COLS, pos="MID", min_rows=100)


def test_returns_model_and_metadata():
    model, algo, params, cv_rho = tune_position_model(
        pos="MID", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf", "xgb"], n_trials=2
    )
    assert algo in ("rf", "xgb")
    assert isinstance(params, dict)
    assert isinstance(cv_rho, float)
    assert hasattr(model, "predict")


def test_single_algo_rf():
    _, algo, _, _ = tune_position_model(
        pos="GK", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf"], n_trials=2
    )
    assert algo == "rf"


def test_single_algo_xgb():
    _, algo, _, _ = tune_position_model(
        pos="FWD", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["xgb"], n_trials=2
    )
    assert algo == "xgb"


def test_model_can_predict():
    model, _, _, _ = tune_position_model(
        pos="DEF", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf", "xgb"], n_trials=2
    )
    preds = model.predict(X_TRAIN)
    assert len(preds) == len(X_TRAIN)


def test_rf_has_oob_score():
    model, algo, _, _ = tune_position_model(
        pos="MID", X_train=X_TRAIN, y_train=Y_TRAIN,
        feat_cols=FEAT_COLS, algos=["rf"], n_trials=2
    )
    if algo == "rf":
        assert hasattr(model, "oob_score_"), "RF must have oob_score_ attribute"
