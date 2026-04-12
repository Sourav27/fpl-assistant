"""Understat client — unique xg_chain + xg_buildup per player per GW (PL only).

Uses soccerdata.Understat (synchronous). Drops all columns that overlap with
FPL API (xg, xa, goals, assists, shots, key_passes, yellow_cards, red_cards).

Season format: 4-digit string "YYXX" where YY = start year, XX = end year.
    "2425" → 2024-25 season
    "2324" → 2023-24 season

Date → GW mapping uses the FPL fixtures API so blank/double GWs are handled
correctly. Multiple fixtures on the same date are resolved by taking the lower
GW number.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Maps 4-digit season codes to human-readable labels. Exported for test assertions.
SEASON_FORMAT_EXAMPLES: dict[str, str] = {
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
}

# Columns from soccerdata Understat that overlap with FPL API — must be dropped.
_FPL_OVERLAP_COLS = frozenset(
    {"xg", "xa", "goals", "assists", "shots", "key_passes", "yellow_cards", "red_cards"}
)

# soccerdata league name for PL
_SD_LEAGUE = "ENG-Premier League"


def _make_understat_reader(season: str):
    """Factory for soccerdata.Understat reader. Isolated for patching in tests."""
    import soccerdata as sd  # type: ignore[import-untyped]

    return sd.Understat(leagues=_SD_LEAGUE, seasons=season)


def _fetch_fixtures_for_season(season: str) -> list[dict[str, Any]]:
    """Fetch FPL fixtures for season to build date → GW map.

    Calls the live FPL fixtures API. Each fixture has at minimum:
        event (int | None)   — GW number (None for unscheduled)
        kickoff_time (str)   — ISO 8601 datetime string
    """
    import urllib.request
    import json

    url = "https://fantasy.premierleague.com/api/fixtures/"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _current_understat_season() -> str:
    """Return the current soccerdata season code, e.g. '2526'. Isolated for patching."""
    import datetime

    now = datetime.date.today()
    # Season starts in August; before August we are still in the previous season
    if now.month >= 8:
        start = now.year
    else:
        start = now.year - 1
    end = start + 1
    return f"{str(start)[2:]}{str(end)[2:]}"


def build_date_gw_map(fixtures: list[dict[str, Any]]) -> dict[str, int]:
    """Build a date → GW mapping from raw FPL fixture dicts.

    Args:
        fixtures: List of FPL fixture dicts with 'event' and 'kickoff_time' fields.

    Returns:
        Dict mapping "YYYY-MM-DD" date strings to GW numbers.
        For dates with multiple fixtures (DGW), the lower GW number is used.
    """
    date_gw: dict[str, int] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        kickoff = fixture.get("kickoff_time")
        if event is None or kickoff is None:
            continue
        date = kickoff[:10]  # "YYYY-MM-DD"
        gw = int(event)
        # On DGW dates, take the lower GW number
        if date not in date_gw or gw < date_gw[date]:
            date_gw[date] = gw
    return date_gw


def fetch_understat_xg_chain(season: str = "2425") -> pd.DataFrame:
    """Return per-player per-GW xg_chain and xg_buildup for the given PL season.

    Args:
        season: 4-digit season code, e.g. "2425" for 2024-25.

    Returns:
        DataFrame with columns: player, team, gw, xg_chain, xg_buildup.
        One row per player per match played.

    Warns:
        If season is not the current season, logs a WARNING about historical
        GW mapping accuracy (FPL fixtures API returns current-season fixtures only).
    """
    current = _current_understat_season()
    if season != current:
        logger.warning(
            "fetch_understat_xg_chain: season '%s' is historical (current: '%s'). "
            "GW mapping uses the live FPL fixtures API which only covers the current "
            "season — historical GW assignment may be inaccurate.",
            season,
            current,
        )

    reader = _make_understat_reader(season)
    raw: pd.DataFrame = reader.read_player_match_stats()

    # soccerdata returns a MultiIndex: (league, season, game, team, player)
    df = raw.reset_index()

    # Extract date from game string e.g. "2024-08-16 Arsenal-Wolves"
    df["match_date"] = df["game"].str[:10]

    # Fetch date → GW map
    fixtures = _fetch_fixtures_for_season(season)
    date_gw = build_date_gw_map(fixtures)

    df["gw"] = df["match_date"].map(date_gw)  # type: ignore[arg-type]

    # Drop FPL-overlapping columns
    cols_to_drop = [c for c in df.columns if c in _FPL_OVERLAP_COLS]
    df = df.drop(columns=cols_to_drop)

    # Drop soccerdata index cols we don't need
    for col in ("league", "season", "game", "match_date"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure the two unique columns we care about are present
    for required in ("xg_chain", "xg_buildup"):
        if required not in df.columns:
            df[required] = 0.0

    return df[["player", "team", "gw", "xg_chain", "xg_buildup"]]  # type: ignore[return-value]
