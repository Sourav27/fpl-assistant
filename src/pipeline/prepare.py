"""Build the merged multi-season dataset from vaastav data + live API patches."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import VAASTAV_DIR, SEASONS, CURRENT_SEASON


LATIN1_SEASONS = {"2016-17", "2017-18", "2018-19"}


def _build_code_map(season: str, vaastav_dir: Path) -> dict:
    """Return {seasonal_element_id: global_player_code} from players_raw.csv.

    FPL's ``code`` field is stable across seasons; ``id`` (element) is recycled
    each season.  Returns an empty dict when players_raw.csv is absent.
    """
    path = vaastav_dir / "data" / season / "players_raw.csv"
    if not path.exists():
        return {}
    raw = pd.read_csv(path, usecols=["id", "code"])
    return dict(zip(raw["id"], raw["code"]))


def load_season_gw_data(season: str, vaastav_dir: Path = VAASTAV_DIR) -> pd.DataFrame:
    """Load merged_gw.csv for a single season."""
    path = vaastav_dir / "data" / season / "gws" / "merged_gw.csv"
    encoding = "latin-1" if season in LATIN1_SEASONS else "utf-8"
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    df["season"] = season

    # Attach persistent player code so features can group by player across seasons.
    # Falls back to element when players_raw.csv is absent (older seasons).
    if "element" in df.columns:
        code_map = _build_code_map(season, vaastav_dir)
        df["code"] = df["element"].map(code_map) if code_map else df["element"]
    return df


def load_live_gw_files(gw_dir: Path) -> pd.DataFrame:
    """Load all gw{N}_live.csv files from a directory."""
    live_files = sorted(gw_dir.glob("gw*_live.csv"))
    if not live_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in live_files]
    return pd.concat(dfs, ignore_index=True, sort=False)


def merge_seasons(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate season DataFrames, taking the union of all columns."""
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def add_fixture_difficulty(gw_df: pd.DataFrame, fixtures_path: Path) -> pd.DataFrame:
    """Join FDR ratings from fixtures.csv onto gameweek data."""
    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures[["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]]
    fixtures = fixtures.rename(columns={"id": "fixture"})

    df = gw_df.merge(fixtures, on="fixture", how="left")

    home = df["was_home"].astype(bool)
    df["fdr_team"] = np.where(home, df["team_h_difficulty"], df["team_a_difficulty"])
    df["fdr_opp"] = np.where(home, df["team_a_difficulty"], df["team_h_difficulty"])
    df = df.drop(columns=["team_h", "team_a", "team_h_difficulty", "team_a_difficulty"])
    return df


def _compute_team_defensive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level defensive rolling stats for opponent join.

    Returns a team-season-GW indexed DataFrame with:
      - team_gc_roll_4: 4-GW lagged rolling avg goals conceded
      - team_pts_allowed_{pos}_roll_6: 6-GW lagged rolling avg points allowed to each position

    Uses shift(1) to prevent lookahead: GW N's stat uses GW 1..(N-1).
    """
    if df.empty or "goals_conceded" not in df.columns:
        return pd.DataFrame(columns=["team", "season", "GW", "team_gc_roll_4"])

    team_gw = (
        df.groupby(["team", "season", "GW"], as_index=False)
        .agg(team_gc=("goals_conceded", "sum"))
    )
    team_gw = team_gw.sort_values(["team", "season", "GW"])
    team_gw["team_gc_roll_4"] = (
        team_gw.groupby(["team", "season"])["team_gc"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    )

    # Points allowed per position: average total_points of opponent players by position
    if "position" in df.columns:
        for pos_label in [1, 2, 3, 4]:
            pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            col = f"team_pts_allowed_{pos_map[pos_label]}_roll_6"
            pos_df = df[df["position"] == pos_label].copy()
            if pos_df.empty:
                team_gw[col] = float("nan")
                continue
            opp_agg = (
                pos_df.groupby(["opponent_team", "season", "GW"], as_index=False)
                .agg(pts_allowed=("total_points", "mean"))
                .rename(columns={"opponent_team": "team"})
            )
            opp_agg = opp_agg.sort_values(["team", "season", "GW"])
            opp_agg[col] = (
                opp_agg.groupby(["team", "season"])["pts_allowed"]
                .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
            )
            team_gw = team_gw.merge(opp_agg[["team", "season", "GW", col]], on=["team", "season", "GW"], how="left")

    return team_gw


def add_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Join opponent defensive stats onto each player row.

    Adds columns:
      - xGC_rolling_4: rolling goals conceded by the OPPONENT team
      - opponent_form_rolling_6: avg pts allowed by opponent to this player's position (6-GW rolling)
    """
    if df.empty or "opponent_team" not in df.columns:
        df["xGC_rolling_4"] = float("nan")
        df["opponent_form_rolling_6"] = float("nan")
        return df

    team_stats = _compute_team_defensive_stats(df)
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    df = df.merge(
        team_stats[["team", "season", "GW", "team_gc_roll_4"]],
        left_on=["opponent_team", "season", "GW"],
        right_on=["team", "season", "GW"],
        how="left",
        suffixes=("", "_opp"),
    )
    df = df.rename(columns={"team_gc_roll_4": "xGC_rolling_4"})
    df = df.drop(columns=["team_opp"], errors="ignore")

    if "position" in df.columns:
        df["_pos_label"] = df["position"].map(pos_map) if df["position"].dtype == object else df["position"].map(pos_map)
        df["opponent_form_rolling_6"] = float("nan")
        for pos_label, pos_str in pos_map.items():
            col = f"team_pts_allowed_{pos_str}_roll_6"
            if col not in team_stats.columns:
                continue
            pos_mask = df["position"] == pos_label
            if not pos_mask.any():
                continue
            temp = df[pos_mask].merge(
                team_stats[["team", "season", "GW", col]],
                left_on=["opponent_team", "season", "GW"],
                right_on=["team", "season", "GW"],
                how="left",
                suffixes=("", "_opp2"),
            )
            df.loc[pos_mask, "opponent_form_rolling_6"] = temp[col].values
        df = df.drop(columns=["_pos_label"], errors="ignore")
    else:
        df["opponent_form_rolling_6"] = float("nan")

    return df


def build_merged_dataset(
    seasons: list[str] | None = None,
    vaastav_dir: Path = VAASTAV_DIR,
) -> pd.DataFrame:
    """Build the full merged dataset: vaastav base + live API patches.

    Deduplication: prefer vaastav over live (richer columns).
    Live data only used for GWs not covered in vaastav's merged_gw.csv.
    """
    seasons = seasons or SEASONS
    dfs = []
    for season in seasons:
        path = vaastav_dir / "data" / season / "gws" / "merged_gw.csv"
        if not path.exists():
            continue
        df = load_season_gw_data(season, vaastav_dir)

        # Load live patches for current season only
        gw_dir = vaastav_dir / "data" / season / "gws"
        live_df = load_live_gw_files(gw_dir) if season == CURRENT_SEASON else pd.DataFrame()
        if not live_df.empty:
            # Only keep live rows for GWs not already in vaastav
            vaastav_gws = set(df["GW"].unique()) if "GW" in df.columns else set()
            live_df = live_df[~live_df["GW"].isin(vaastav_gws)]
            if not live_df.empty:
                live_df["season"] = season
                df = pd.concat([df, live_df], ignore_index=True, sort=False)

        fixtures_path = vaastav_dir / "data" / season / "fixtures.csv"
        if fixtures_path.exists():
            df = add_fixture_difficulty(df, fixtures_path)

        # B-F1/B-F2: join opponent defensive stats
        if not df.empty and "opponent_team" in df.columns:
            df = add_opponent_stats(df)

        dfs.append(df)

    return merge_seasons(dfs)
