import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.pipeline.predict import compute_shap_reasons

RNG = np.random.default_rng(0)
N = 100
FEAT_COLS = ["minutes_roll_4", "ict_index_roll_4", "xGC_rolling_4", "is_home", "transfers_net"]
X = pd.DataFrame(RNG.random((N, len(FEAT_COLS))), columns=FEAT_COLS)
Y = pd.Series(RNG.integers(0, 12, N).astype(float))


@pytest.fixture
def rf_model():
    m = RandomForestRegressor(n_estimators=10, random_state=0)
    m.fit(X, Y)
    return m


@pytest.fixture
def xgb_model():
    m = XGBRegressor(n_estimators=10, random_state=0, verbosity=0)
    m.fit(X, Y)
    return m


def test_returns_series_same_length(rf_model):
    result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=5)
    assert len(result) == N


def test_reason_string_format(rf_model):
    """Format: 'feat_name val (rank R/N) | feat_name val (rank R/N) | ...'"""
    result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=3)
    sample = result.iloc[0]
    assert isinstance(sample, str)
    parts = [p.strip() for p in sample.split("|")]
    assert len(parts) == 3
    for part in parts:
        assert "(rank " in part, f"Missing rank in part: {part!r}"
        assert "/100)" in part, f"Expected /N) in part: {part!r}"


def test_rank_within_cohort(rf_model):
    """Rank should be within cohort_X (full position set), not just X."""
    cohort = pd.concat([X, X + 0.5], ignore_index=True)  # 200-row cohort
    result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=1, cohort_X=cohort)
    # Rank denominator should be cohort size (200), not X size (100)
    assert "/200)" in result.iloc[0], f"Expected /200) in: {result.iloc[0]!r}"


def test_works_with_xgb(xgb_model):
    result = compute_shap_reasons(xgb_model, X, FEAT_COLS, top_n=2)
    assert len(result) == N
    assert result.iloc[0].count("|") == 1  # 2 parts → 1 pipe


def test_top_n_respected(rf_model):
    for top_n in [1, 2, 5]:
        result = compute_shap_reasons(rf_model, X, FEAT_COLS, top_n=top_n)
        parts = result.iloc[0].split("|")
        assert len(parts) == top_n


def test_raises_on_column_mismatch(rf_model):
    """Feature column mismatch must raise, not silently produce wrong SHAP labels."""
    wrong_cols = ["col_a", "col_b", "col_c", "col_d", "col_e"]
    X_wrong = pd.DataFrame(RNG.random((N, 5)), columns=wrong_cols)
    with pytest.raises(ValueError, match="column mismatch"):
        compute_shap_reasons(rf_model, X_wrong, wrong_cols, top_n=3)
