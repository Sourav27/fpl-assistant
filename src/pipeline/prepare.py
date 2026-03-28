"""Build the merged multi-season dataset from vaastav data + live API patches."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import VAASTAV_DIR, SEASONS, CURRENT_SEASON


LATIN1_SEASONS = {"2016-17", "2017-18", "2018-19"}


def load_season_gw_data(season: str, vaastav_dir: Path = VAASTAV_DIR) -> pd.DataFrame:
    """Load merged_gw.csv for a single season."""
    path = vaastav_dir / "data" / season / "gws" / "merged_gw.csv"
    encoding = "latin-1" if season in LATIN1_SEASONS else "utf-8"
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    df["season"] = season
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
        dfs.append(df)

    return merge_seasons(dfs)
