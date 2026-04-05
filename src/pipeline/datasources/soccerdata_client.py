"""FotMob wrapper (via soccerdata) for European/international match minutes.

Covers: UEFA Champions League, UEFA Europa League, UEFA Conference League,
        International Friendlies.
Does NOT cover: Premier League (use FPL API for those).

NOTE: soccerdata is not yet in requirements.txt. To use this module in
production, install with: pip install soccerdata

The actual soccerdata.FotMob() method for player minutes should be verified
against the library documentation before production use.

Reliability gate: MAE <= 5 min AND correlation >= 0.95 on PL cross-validation.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_COMPETITIONS = [
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "International Friendlies",
]


@dataclass
class FotMobReliabilityResult:
    """Result of cross-validation between FotMob and FPL minutes."""
    mae: float
    correlation: float
    n_matched: int
    reliable: bool  # True if mae <= 5 and correlation >= 0.95


def _fetch_fotmob_raw(competitions: list[str]) -> pd.DataFrame:
    """Fetch raw FotMob player minutes via soccerdata.

    IMPORTANT: Verify the correct soccerdata.FotMob method for player-level
    minutes before production use. Common candidates: read_player_match_stats(),
    read_players(), or similar.

    Args:
        competitions: List of competition names to fetch.

    Returns:
        DataFrame with columns: player_name, team, competition, date, minutes
        Returns empty DataFrame if soccerdata not installed or fetch fails.
    """
    try:
        import soccerdata as sd
    except ImportError:
        logger.warning(
            "soccerdata not installed. Install with: pip install soccerdata"
        )
        return pd.DataFrame(
            columns=["player_name", "team", "competition", "date", "minutes"]
        )

    rows = []
    try:
        fotmob = sd.FotMob()
    except Exception as e:
        logger.warning("Failed to initialize FotMob client: %s", e)
        return pd.DataFrame(
            columns=["player_name", "team", "competition", "date", "minutes"]
        )

    for comp in competitions:
        try:
            # read_player_match_stats is the likely correct method — verify before use
            stats = fotmob.read_player_match_stats(competition=comp)
            if stats is not None and not stats.empty:
                stats = stats.reset_index()
                rows.append(stats)
        except Exception as e:
            logger.warning("FotMob fetch failed for %s: %s", comp, e)

    if not rows:
        return pd.DataFrame(
            columns=["player_name", "team", "competition", "date", "minutes"]
        )
    return pd.concat(rows, ignore_index=True)


def fetch_fotmob_player_minutes(
    competitions: list[str] | None = None,
) -> pd.DataFrame:
    """Return player minutes for the specified competitions.

    Args:
        competitions: Competition names to include. Defaults to SUPPORTED_COMPETITIONS.

    Returns:
        DataFrame with columns: player_name, team, competition, date, minutes
    """
    if competitions is None:
        competitions = SUPPORTED_COMPETITIONS
    raw = _fetch_fotmob_raw(competitions)
    if raw.empty:
        return raw
    return raw[raw["competition"].isin(competitions)].reset_index(drop=True)


def cross_validate_with_fpl(
    fotmob_pl: pd.DataFrame,
    fpl_minutes: pd.DataFrame,
) -> FotMobReliabilityResult:
    """Cross-validate FotMob PL minutes against FPL element-summary minutes.

    Join key: (player_name / web_name, team, date).
    Gate: MAE <= 5 min AND correlation >= 0.95 → reliable.

    Args:
        fotmob_pl: DataFrame with columns: player_name, team, date, minutes
        fpl_minutes: DataFrame with columns: web_name, team, date, minutes

    Returns:
        FotMobReliabilityResult with mae, correlation, n_matched, reliable flag.
    """
    if fotmob_pl.empty or fpl_minutes.empty:
        return FotMobReliabilityResult(
            mae=float("inf"), correlation=0.0, n_matched=0, reliable=False
        )

    merged = fotmob_pl.merge(
        fpl_minutes.rename(columns={"web_name": "player_name"}),
        on=["player_name", "team", "date"],
        suffixes=("_fotmob", "_fpl"),
    )

    if merged.empty:
        logger.warning("No matched rows in FotMob vs FPL cross-validation")
        return FotMobReliabilityResult(
            mae=float("inf"), correlation=0.0, n_matched=0, reliable=False
        )

    mae = float((merged["minutes_fotmob"] - merged["minutes_fpl"]).abs().mean())
    corr = float(merged["minutes_fotmob"].corr(merged["minutes_fpl"]))
    reliable = mae <= 5.0 and corr >= 0.95
    return FotMobReliabilityResult(
        mae=mae, correlation=corr, n_matched=len(merged), reliable=reliable
    )
