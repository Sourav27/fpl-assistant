import pytest
import pandas as pd
from src.pipeline.source_validation import (
    compute_source_spearman,
    run_xg_validation_gate,
    SourceValidationResult,
)


def _make_df(xg_col: str, xg_vals, goals_vals) -> pd.DataFrame:
    return pd.DataFrame({xg_col: xg_vals, "goals_scored": goals_vals})


def test_spearman_perfect_correlation():
    df = _make_df("understat_xG", [0.1, 0.3, 0.7, 1.2], [0, 0, 1, 2])
    rho = compute_source_spearman(df, xg_col="understat_xG", actual_col="goals_scored")
    assert rho > 0.9


def test_spearman_returns_float():
    df = _make_df("xG", [0.2, 0.5, 0.9], [0, 1, 1])
    rho = compute_source_spearman(df, xg_col="xG", actual_col="goals_scored")
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0


def test_gate_passes_when_understat_within_tolerance():
    result = run_xg_validation_gate(
        understat_rho=0.62,
        fpl_opta_rho=0.65,
        tolerance=0.05,
    )
    assert result.passed is True
    assert result.recommended_source == "understat"


def test_gate_fails_when_understat_too_low():
    result = run_xg_validation_gate(
        understat_rho=0.55,
        fpl_opta_rho=0.65,
        tolerance=0.05,
    )
    assert result.passed is False
    assert result.recommended_source == "vaastav_goals_conceded"


def test_gate_result_has_required_fields():
    result = run_xg_validation_gate(0.68, 0.65, 0.05)
    assert hasattr(result, "passed")
    assert hasattr(result, "understat_rho")
    assert hasattr(result, "fpl_opta_rho")
    assert hasattr(result, "recommended_source")
