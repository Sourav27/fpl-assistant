# src/pipeline/features.py
"""Vectorized feature engineering — replaces NB02 iterrows approach."""
import pandas as pd

ROLLING_COLS = [
    "total_points", "minutes", "ict_index", "bps",
    "goals_scored", "assists", "clean_sheets",
    "influence", "creativity", "threat",
]

DEFAULT_WINDOWS = [4, 8]


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add lagged rolling averages per player.

    Uses shift(1) so GW N's features are computed from GW 1..(N-1).
    """
    windows = windows or DEFAULT_WINDOWS
    cols = cols or [c for c in ROLLING_COLS if c in df.columns]
    if df.empty:
        return df
    # Use persistent player code when available; fall back to seasonal element ID.
    player_id = "code" if "code" in df.columns else "element"
    df = df.sort_values([player_id, "season", "GW"]).copy()

    for col in cols:
        for w in windows:
            df[f"{col}_roll_{w}"] = (
                df.groupby([player_id, "season"])[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum = short-term rolling avg - long-term rolling avg."""
    for col in ROLLING_COLS:
        short = f"{col}_roll_4"
        long = f"{col}_roll_8"
        if short in df.columns and long in df.columns:
            df[f"{col}_momentum"] = df[short] - df[long]
    return df


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived form features."""
    if "transfers_in" in df.columns and "transfers_out" in df.columns:
        df["transfers_net"] = df["transfers_in"] - df["transfers_out"]
    return df


def engineer_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_rolling_features(df)
    df = add_momentum_features(df)
    df = add_form_features(df)
    if drop_na:
        longest_window = max(DEFAULT_WINDOWS)
        roll_col = f"total_points_roll_{longest_window}"
        if roll_col in df.columns:
            df = df.dropna(subset=[roll_col])
    return df
