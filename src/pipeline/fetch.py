"""FPL API data fetching — thin wrappers with retry logic and live data collection."""
import time
import logging
import pandas as pd
import requests
from src.config import (
    FPL_BOOTSTRAP_URL, FPL_PLAYER_URL, FPL_FIXTURES_URL,
    CURRENT_SEASON, API_REQUEST_DELAY, API_RETRY_ATTEMPTS, API_RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)

# element_type ID → position string (API uses "GKP" but we normalize to "GK" for vaastav compat)
ELEMENT_TYPE_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Columns that exist in vaastav but cannot be derived from API
UNAVAILABLE_FROM_API = [
    "clearances_blocks_interceptions", "defensive_contribution",
    "recoveries", "tackles",
]


def _api_get_with_retry(url: str, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff retry."""
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == API_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(API_RETRY_BASE_DELAY * (2 ** attempt))


def fetch_bootstrap() -> dict:
    """Fetch the main FPL bootstrap-static endpoint."""
    return _api_get_with_retry(FPL_BOOTSTRAP_URL).json()


def fetch_player_history(player_id: int) -> dict:
    """Fetch individual player GW history + past seasons."""
    url = f"{FPL_PLAYER_URL}/{player_id}/"
    return _api_get_with_retry(url, timeout=15).json()


def fetch_fixtures() -> list[dict]:
    """Fetch all fixtures for the season."""
    return _api_get_with_retry(FPL_FIXTURES_URL).json()


def get_current_gw(bootstrap_data: dict) -> int | None:
    """Return the current gameweek number from bootstrap data."""
    for event in bootstrap_data["events"]:
        if event["is_current"]:
            return event["id"]
    return None


def get_next_deadline(bootstrap_data: dict) -> tuple[int, str]:
    """Return (gw_number, deadline_time) for the next upcoming GW."""
    for event in bootstrap_data["events"]:
        if event["is_next"]:
            return event["id"], event["deadline_time"]
    raise ValueError("No upcoming gameweek found")


def extract_xp_snapshot(bootstrap_data: dict) -> dict[int, float]:
    """Extract {player_id: ep_this} from bootstrap data.

    Must be called BEFORE the GW deadline — ep_this is forward-looking.
    """
    result = {}
    for el in bootstrap_data["elements"]:
        ep = el.get("ep_this")
        result[el["id"]] = float(ep) if ep is not None else 0.0
    return result


def _build_bootstrap_lookups(bootstrap_data: dict) -> tuple[dict, dict, dict]:
    """Build lookup dicts from bootstrap data for normalization."""
    team_map = {t["id"]: t["name"] for t in bootstrap_data["teams"]}
    element_map = {e["id"]: e for e in bootstrap_data["elements"]}
    pos_map = ELEMENT_TYPE_MAP.copy()
    return team_map, element_map, pos_map


def normalize_player_gw_to_vaastav(
    gw_row: dict,
    bootstrap_data: dict,
    _lookups: tuple | None = None,
) -> dict:
    """Normalize a single API player-GW history row to vaastav schema."""
    team_map, element_map, pos_map = _lookups or _build_bootstrap_lookups(bootstrap_data)
    element_id = gw_row["element"]
    element_info = element_map.get(element_id, {})

    row = {
        # Derived fields
        "name": element_info.get("web_name", "Unknown"),
        "position": pos_map.get(element_info.get("element_type"), "UNK"),
        "team": team_map.get(element_info.get("team"), "Unknown"),
        "element": element_id,
        "GW": gw_row["round"],
        "season": CURRENT_SEASON,
        "xP": 0.0,  # filled later from xP snapshot if available
        # Directly mapped fields
        "total_points": gw_row.get("total_points", 0),
        "minutes": gw_row.get("minutes", 0),
        "goals_scored": gw_row.get("goals_scored", 0),
        "assists": gw_row.get("assists", 0),
        "clean_sheets": gw_row.get("clean_sheets", 0),
        "goals_conceded": gw_row.get("goals_conceded", 0),
        "bonus": gw_row.get("bonus", 0),
        "bps": gw_row.get("bps", 0),
        "influence": float(gw_row.get("influence", 0)),
        "creativity": float(gw_row.get("creativity", 0)),
        "threat": float(gw_row.get("threat", 0)),
        "ict_index": float(gw_row.get("ict_index", 0)),
        "value": gw_row.get("value", 0),
        "transfers_in": gw_row.get("transfers_in", 0),
        "transfers_out": gw_row.get("transfers_out", 0),
        "selected": gw_row.get("selected", 0),
        "was_home": gw_row.get("was_home", False),
        "opponent_team": gw_row.get("opponent_team", 0),
        "fixture": gw_row.get("fixture", 0),
        "round": gw_row.get("round", 0),
        "kickoff_time": gw_row.get("kickoff_time", ""),
        "starts": gw_row.get("starts", 0),
        "expected_goals": float(gw_row.get("expected_goals", 0)),
        "expected_assists": float(gw_row.get("expected_assists", 0)),
        "expected_goal_involvements": float(gw_row.get("expected_goal_involvements", 0)),
        "expected_goals_conceded": float(gw_row.get("expected_goals_conceded", 0)),
    }

    # Unavailable from API — fill with NaN
    for col in UNAVAILABLE_FROM_API:
        row[col] = float("nan")

    return row


def fetch_live_gw_data(
    target_gw: int,
    bootstrap_data: dict,
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Fetch all player data for a specific GW and normalize to vaastav schema.

    Args:
        target_gw: The GW number to collect data for.
        bootstrap_data: Bootstrap-static response (for lookups).
        player_ids: Optional subset of player IDs. Defaults to all active players.
    """
    if player_ids is None:
        player_ids = [e["id"] for e in bootstrap_data["elements"]]

    lookups = _build_bootstrap_lookups(bootstrap_data)

    rows = []
    for i, pid in enumerate(player_ids):
        if i > 0:
            time.sleep(API_REQUEST_DELAY)
        if (i + 1) % 50 == 0:
            logger.info(f"Fetching player {i + 1}/{len(player_ids)}")

        try:
            data = fetch_player_history(pid)
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch player {pid}: {e}")
            continue

        for gw_row in data.get("history", []):
            if gw_row["round"] == target_gw:
                normalized = normalize_player_gw_to_vaastav(gw_row, bootstrap_data, _lookups=lookups)
                rows.append(normalized)
                break

    return pd.DataFrame(rows) if rows else pd.DataFrame()
