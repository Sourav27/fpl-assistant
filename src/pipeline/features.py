# src/pipeline/features.py
"""Vectorized feature engineering — replaces NB02 iterrows approach."""
import pandas as pd

FIXTURE_FEATURE_COLUMNS = [
    "xGC_rolling_4",
    "opponent_form_rolling_6",
    "is_home",
    "fixture_count",
    "rest_days",
    "is_fixture_2",
]

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


def add_fixture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-fixture context features.

    Expects df to already have one row per fixture (DGW players appear twice).
    - is_home: 1/0 from was_home
    - fixture_count: how many fixtures this player has in this GW (1=normal, 2=DGW)
    - is_fixture_2: 1 for the second game in a DGW, else 0
    - rest_days: days between fixture 1 and fixture 2 (0 for fixture 1 and non-DGW)
    """
    df = df.copy()

    # is_home
    if "was_home" in df.columns:
        df["is_home"] = df["was_home"].astype(int)
    else:
        df["is_home"] = 0

    # fixture_count: count of rows per player per GW
    player_id = "code" if "code" in df.columns else "element"
    df["fixture_count"] = df.groupby([player_id, "season", "GW"])["GW"].transform("count")

    # Sort within each player-GW group by kickoff time to identify fixture order
    if "kickoff_time" in df.columns:
        df["_kickoff_dt"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        df = df.sort_values([player_id, "season", "GW", "_kickoff_dt"])
        df["_fixture_rank"] = df.groupby([player_id, "season", "GW"]).cumcount()
        df["is_fixture_2"] = (df["_fixture_rank"] == 1).astype(int)

        # rest_days for fixture 2 = days since fixture 1
        first_kickoffs = (
            df[df["_fixture_rank"] == 0]
            .set_index([player_id, "season", "GW"])["_kickoff_dt"]
        )
        df = df.join(first_kickoffs.rename("_first_ko"), on=[player_id, "season", "GW"])
        df["rest_days"] = 0.0
        mask = df["is_fixture_2"] == 1
        df.loc[mask, "rest_days"] = (
            (df.loc[mask, "_kickoff_dt"] - df.loc[mask, "_first_ko"])
            .dt.total_seconds() / 86400
        ).clip(lower=0)
        df = df.drop(columns=["_kickoff_dt", "_fixture_rank", "_first_ko"], errors="ignore")
    else:
        df["is_fixture_2"] = 0
        df["rest_days"] = 0.0

    return df


def engineer_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_rolling_features(df)
    df = add_momentum_features(df)
    df = add_form_features(df)
    df = add_fixture_features(df)
    if drop_na:
        longest_window = max(DEFAULT_WINDOWS)
        roll_col = f"total_points_roll_{longest_window}"
        if roll_col in df.columns:
            df = df.dropna(subset=[roll_col])
    return df
