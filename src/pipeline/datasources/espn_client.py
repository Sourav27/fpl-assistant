"""ESPN direct API client — non-PL competition minutes per player.

Covers: UCL, UEL, UECL, FA Cup, Carabao Cup, FIFA Friendlies, Internationals.
PL matches (eng.1) are excluded — covered by FPL API.

Public API:
    resolve_espn_player_id(fpl_code, web_name, second_name) -> int | None
    fetch_espn_player_season(espn_id, season_year) -> pd.DataFrame
    fetch_espn_recent(espn_id, days) -> pd.DataFrame

Rate limiting: 1s sleep between player fetches. 429 → exponential backoff
(2s, 4s, 8s, max 3 retries). Per-player-season CSV cache; re-runs skip
already-cached players.
"""
from __future__ import annotations

import csv
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parents[3] / "data"
_RESULTS_DIR = Path(__file__).parents[3] / "results"
_ID_MAP_PATH = _DATA_DIR / "espn_player_id_map.csv"
_CACHE_DIR = _RESULTS_DIR / "espn_cache"
_UNRESOLVED_PATH = _RESULTS_DIR / "espn_unresolved.csv"

# ── ESPN API ───────────────────────────────────────────────────────────────────
_EVENTLOG_URL = (
    "https://sports.core.api.espn.com/v2/sports/soccer"
    "/athletes/{espn_id}/eventlog?season={season_year}&limit=200"
)
_EVENT_SUMMARY_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/soccer/{league}/summary"
    "?event={event_id}"
)

# PL league slug — excluded (covered by FPL API)
_PL_SLUG = "eng.1"

# Fuzzy match confidence threshold
_FUZZY_THRESHOLD = 0.85

# Columns emitted per match
_OUTPUT_COLS = [
    "espn_id",
    "fpl_code",
    "match_date",
    "league_slug",
    "competition",
    "minutes",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "yellow_cards",
    "red_cards",
    "fouls_committed",
    "fouls_suffered",
    "offsides",
]


# ── ID resolution ──────────────────────────────────────────────────────────────

def _load_id_map() -> dict[int, dict[str, Any]]:
    """Load espn_player_id_map.csv into {fpl_code: row_dict}."""
    id_map: dict[int, dict[str, Any]] = {}
    if not _ID_MAP_PATH.exists():
        return id_map
    with _ID_MAP_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                id_map[int(row["fpl_code"])] = row
            except (KeyError, ValueError):
                pass
    return id_map


def resolve_espn_player_id(
    fpl_code: int,
    web_name: str,
    second_name: str,
) -> int | None:
    """Resolve an FPL player code to an ESPN athlete ID.

    Strategy:
    1. Look up fpl_code in the seeded espn_player_id_map.csv.
    2. If not found, try fuzzy name match against the seeded entries with
       confidence ≥ _FUZZY_THRESHOLD (0.85).
    3. If still unresolved, log to results/espn_unresolved.csv and return None.

    Args:
        fpl_code:    Persistent FPL player code (cross-season identifier).
        web_name:    FPL web_name (e.g. "Saka").
        second_name: FPL second_name (e.g. "Saka").

    Returns:
        ESPN athlete int ID, or None if unresolvable.
    """
    id_map = _load_id_map()

    # 1. Direct lookup
    if fpl_code in id_map:
        return int(id_map[fpl_code]["espn_id"])

    # 2. Fuzzy name fallback
    best_score = 0.0
    best_espn_id: int | None = None
    query = f"{web_name} {second_name}".lower().strip()

    for row in id_map.values():
        candidate = row.get("espn_name", "").lower()
        score = _name_similarity(query, candidate)
        if score > best_score:
            best_score = score
            best_espn_id = int(row["espn_id"])

    if best_score >= _FUZZY_THRESHOLD and best_espn_id is not None:
        logger.info(
            "ESPN ID fuzzy match: fpl_code=%d '%s' → espn_id=%d (score=%.2f)",
            fpl_code,
            web_name,
            best_espn_id,
            best_score,
        )
        return best_espn_id

    # 3. Log unresolved
    _log_unresolved(fpl_code, web_name, second_name)
    return None


def _name_similarity(a: str, b: str) -> float:
    """Jaccard similarity on character bigrams — fast, no external deps."""
    def bigrams(s: str) -> set[str]:
        s = s.replace(" ", "")
        return {s[i : i + 2] for i in range(len(s) - 1)}

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a and not bg_b:
        return 1.0
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def _log_unresolved(fpl_code: int, web_name: str, second_name: str) -> None:
    _UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _UNRESOLVED_PATH.exists()
    with _UNRESOLVED_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fpl_code", "web_name", "second_name"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"fpl_code": fpl_code, "web_name": web_name, "second_name": second_name}
        )
    logger.warning(
        "ESPN ID unresolved: fpl_code=%d '%s' — added to %s",
        fpl_code,
        web_name,
        _UNRESOLVED_PATH,
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_json(url: str, retries: int = 3, backoff: float = 2.0) -> dict | list | None:
    """Fetch JSON from URL with exponential backoff on 429."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                delay = backoff * (2 ** attempt)
                logger.warning("ESPN 429 rate limit — sleeping %.1fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
            else:
                logger.error("ESPN HTTP %d fetching %s", exc.code, url)
                return None
        except Exception as exc:
            logger.error("ESPN fetch error for %s: %s", url, exc)
            return None
    return None


# ── Season fetch ───────────────────────────────────────────────────────────────

def fetch_espn_player_season(
    espn_id: int,
    season_year: int,
    fpl_code: int | None = None,
) -> pd.DataFrame:
    """Fetch all non-PL match stats for a player in a given season.

    Results are cached at results/espn_cache/player_{espn_id}_season_{year}.csv.
    Re-runs skip players with an existing cache file.

    Args:
        espn_id:     ESPN athlete ID.
        season_year: 4-digit year (e.g. 2024 for 2024-25 season).
        fpl_code:    FPL player code, stored in the cache for join purposes.

    Returns:
        DataFrame with columns matching _OUTPUT_COLS. Empty DataFrame if no
        non-PL matches found or fetch fails.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"player_{espn_id}_season_{season_year}.csv"

    if cache_path.exists():
        logger.debug("ESPN cache hit: %s", cache_path)
        return pd.read_csv(cache_path)

    url = _EVENTLOG_URL.format(espn_id=espn_id, season_year=season_year)
    data = _get_json(url)
    if not data:
        return _empty_df()

    events = _extract_events(data)
    rows = []

    for event in events:
        league_slug = event.get("league_slug", "")
        if league_slug == _PL_SLUG:
            continue  # PL covered by FPL API

        stats = _fetch_event_stats(espn_id, event)
        if stats is None:
            continue

        rows.append(
            {
                "espn_id": espn_id,
                "fpl_code": fpl_code,
                "match_date": event.get("date", ""),
                "league_slug": league_slug,
                "competition": event.get("competition", ""),
                **stats,
            }
        )
        time.sleep(1.0)

    df = pd.DataFrame(rows, columns=_OUTPUT_COLS) if rows else _empty_df()
    df.to_csv(cache_path, index=False)
    logger.info(
        "ESPN fetched espn_id=%d season=%d → %d non-PL matches (cached)",
        espn_id,
        season_year,
        len(df),
    )
    return df


def fetch_espn_recent(espn_id: int, days: int = 30) -> pd.DataFrame:
    """Fetch non-PL matches for an ESPN player in the last N days.

    Used in the weekly prediction run for fatigue signal (non_pl_minutes_roll_4).

    Args:
        espn_id: ESPN athlete ID.
        days:    Look-back window in days (default 30).

    Returns:
        DataFrame with same columns as fetch_espn_player_season.
    """
    cutoff = date.today() - timedelta(days=days)
    season_year = cutoff.year if cutoff.month >= 8 else cutoff.year - 1

    df = fetch_espn_player_season(espn_id, season_year)
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    cutoff_dt = pd.Timestamp(cutoff)
    return df[df["match_date"] >= cutoff_dt].reset_index(drop=True)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_events(data: dict | list) -> list[dict]:
    """Extract event list from ESPN eventlog API response."""
    if isinstance(data, list):
        return data

    events = []
    for item in data.get("events", {}).get("items", []):
        ref = item.get("$ref", "")
        # Extract league slug from eventlog ref URL heuristic
        # e.g. ".../soccer/uefa.champions/events/..."
        league_slug = _extract_league_slug(ref)
        event_id = _extract_event_id(ref)
        events.append(
            {
                "league_slug": league_slug,
                "event_id": event_id,
                "date": item.get("date", "")[:10],
                "competition": _slug_to_competition(league_slug),
            }
        )
    return events


def _extract_league_slug(ref: str) -> str:
    """Extract league slug from ESPN $ref URL, e.g. 'uefa.champions'."""
    parts = ref.split("/")
    try:
        idx = parts.index("soccer")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def _extract_event_id(ref: str) -> str:
    """Extract event ID from ESPN $ref URL."""
    parts = ref.rstrip("/").split("/")
    # Last segment is usually the event id
    try:
        idx = parts.index("events")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return parts[-1] if parts else ""


def _slug_to_competition(slug: str) -> str:
    mapping = {
        "uefa.champions": "UCL",
        "uefa.europa": "UEL",
        "uefa.europa.conf": "UECL",
        "eng.fa": "FA_Cup",
        "eng.league_cup": "Carabao",
        "fifa.friendly": "FIFA_Friendly",
    }
    return mapping.get(slug, slug)


def _fetch_event_stats(espn_id: int, event: dict) -> dict | None:
    """Fetch player-level stats for a single ESPN event. Returns stat dict or None."""
    league_slug = event.get("league_slug", "")
    event_id = event.get("event_id", "")
    if not league_slug or not event_id:
        return None

    url = _EVENT_SUMMARY_URL.format(league=league_slug, event_id=event_id)
    data = _get_json(url)
    if not data:
        return None

    return _parse_player_stats(data, espn_id)


def _parse_player_stats(summary: dict, espn_id: int) -> dict | None:
    """Extract per-player stats from an ESPN event summary response."""
    stat_defaults: dict[str, int] = {
        "minutes": 0,
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "shots_on_target": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "fouls_committed": 0,
        "fouls_suffered": 0,
        "offsides": 0,
    }

    # ESPN summary nests player stats under boxscore → players → statistics
    boxscore = summary.get("boxscore", {})
    for team_block in boxscore.get("players", []):
        for stat_block in team_block.get("statistics", []):
            for athlete_block in stat_block.get("athletes", []):
                athlete = athlete_block.get("athlete", {})
                if str(athlete.get("id", "")) == str(espn_id):
                    return _parse_stat_block(athlete_block, stat_defaults)

    return None


def _parse_stat_block(athlete_block: dict, defaults: dict) -> dict:
    """Parse ESPN stat names/values from an athlete statistics block."""
    stats = dict(defaults)
    stat_name_map = {
        "minutesPlayed": "minutes",
        "goals": "goals",
        "assists": "assists",
        "totalShots": "shots",
        "shotsOnTarget": "shots_on_target",
        "yellowCards": "yellow_cards",
        "redCards": "red_cards",
        "foulsCommitted": "fouls_committed",
        "foulsSuffered": "fouls_suffered",
        "offsides": "offsides",
    }
    names = athlete_block.get("names", [])
    values = athlete_block.get("stats", [])
    for name, value in zip(names, values):
        col = stat_name_map.get(name)
        if col:
            try:
                stats[col] = int(float(value))
            except (ValueError, TypeError):
                pass
    return stats


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLS)
