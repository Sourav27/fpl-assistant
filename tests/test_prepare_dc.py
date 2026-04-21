import pandas as pd
import pytest
from src.pipeline.prepare import compute_points_with_dc


def _make_player_df():
    return pd.DataFrame({
        "code": [1, 2, 3, 4],
        "position": ["DEF", "MID", "FWD", "GK"],
        "total_points": [6, 4, 8, 2],
        # DEF: 7+2+0+3=12 >= 10 -> +2
        # MID: 3+1+2+4+1=11 < 12 -> +0
        # FWD: 2+1+3+5+3=14 >= 12 -> +2
        # GK: DC not applicable
        "clearances":      [7, 3, 2, 0],
        "blocked_shots":   [2, 1, 1, 0],
        "interceptions":   [0, 2, 3, 0],
        "tackles":         [3, 4, 5, 0],
        "recoveries":      [0, 1, 3, 0],
    })


def test_def_dc_threshold_met():
    df = compute_points_with_dc(_make_player_df())
    assert df.loc[df["code"] == 1, "points_with_DC"].iloc[0] == 8  # 6 + 2


def test_mid_dc_threshold_not_met():
    df = compute_points_with_dc(_make_player_df())
    assert df.loc[df["code"] == 2, "points_with_DC"].iloc[0] == 4  # unchanged


def test_fwd_dc_threshold_met():
    df = compute_points_with_dc(_make_player_df())
    assert df.loc[df["code"] == 3, "points_with_DC"].iloc[0] == 10  # 8 + 2


def test_gk_unaffected():
    df = compute_points_with_dc(_make_player_df())
    assert df.loc[df["code"] == 4, "points_with_DC"].iloc[0] == 2  # unchanged


def test_nan_dc_falls_back_to_total_points():
    """Rows without FBref data (no DC columns) must fall back to total_points."""
    df = _make_player_df().drop(columns=["clearances", "blocked_shots",
                                          "interceptions", "tackles", "recoveries"])
    result = compute_points_with_dc(df)
    assert (result["points_with_DC"] == result["total_points"]).all()
