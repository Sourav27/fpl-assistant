"""understatAPI client — PL xG, xA, xGC per player per GW (EPL only).

understatAPI covers 6 leagues: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL.
It does NOT cover Champions League, Europa League, or international matches.

asyncio note: asyncio.run() raises RuntimeError if called inside an already-running
event loop (e.g. Jupyter, FastAPI). For CLI pipeline use only. In async contexts,
await _fetch_player_grouped_stats_async() directly.

xGC derivation: For each fixture, sum xG of all opponent players against the
defending team. This gives team-level xGC per match.
"""
from __future__ import annotations
import asyncio
import logging
import pandas as pd

logger = logging.getLogger(__name__)

UNDERSTAT_SEASON_MAP = {
    "2025-26": "2025",
    "2024-25": "2024",
    "2023-24": "2023",
    "2022-23": "2022",
    "2021-22": "2021",
}


async def _fetch_player_grouped_stats_async(season: str) -> list[dict]:
    """Async inner — fetch per-player per-match stats from understatAPI.

    Each entry contains: player_id, player, team, xG, xA, time, date,
    id (fixture), h_team, a_team, and other understat fields.
    """
    from understatapi import UnderstatClient
    async with UnderstatClient() as client:
        data = await client.league(league="EPL").get_player_data(season=season)
    return data


def fetch_understat_player_gw_stats(season: str = "2025") -> pd.DataFrame:
    """Return a DataFrame of per-player per-match xG/xA stats for the given season.

    Args:
        season: Understat season string, e.g. "2025" for 2025-26.

    Columns: player_id, player, team, xG (float), xA (float), time (int),
             date (str), fixture_id (str), h_team, a_team
    """
    raw = asyncio.run(_fetch_player_grouped_stats_async(season))
    df = pd.DataFrame(raw)
    df = df.rename(columns={"id": "fixture_id"})
    for col in ("xG", "xA"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
    return df


def compute_team_xgc_per_gw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level xG created per fixture from player-level xG rows.

    For each fixture, a team's xGC = sum of xG from all players on that team.
    To find what a team conceded, look at the opposing team's row in the same
    fixture_id group.

    Returns DataFrame with columns: fixture_id, team, xGC
    """
    if df.empty:
        return pd.DataFrame(columns=["fixture_id", "team", "xGC"])

    # Accept either the raw "id" column (pre-rename) or the canonical "fixture_id"
    if "fixture_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "fixture_id"})

    rows = []
    for fixture_id, group in df.groupby("fixture_id"):
        h_team = group["h_team"].iloc[0]
        a_team = group["a_team"].iloc[0]
        # xGC per team = sum of xG created by that team's players in this fixture.
        # Downstream, a team's xGC represents how threatening their attack was;
        # the opponent's row holds what they conceded.
        home_xg = group[group["team"] == h_team]["xG"].sum()
        away_xg = group[group["team"] == a_team]["xG"].sum()
        rows.append({"fixture_id": fixture_id, "team": h_team, "xGC": home_xg})
        rows.append({"fixture_id": fixture_id, "team": a_team, "xGC": away_xg})

    return pd.DataFrame(rows)
