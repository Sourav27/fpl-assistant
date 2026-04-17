"""FPL API data fetching — thin wrappers with retry logic and live data collection."""
import time
import logging
import pandas as pd
import requests
from src.config import (
    FPL_BOOTSTRAP_URL, FPL_PLAYER_URL, FPL_FIXTURES_URL, FPL_EVENT_URL,
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


WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"
WAYBACK_LOOKBACK_DAYS = 7


def fetch_bootstrap() -> dict:
    """Fetch the main FPL bootstrap-static endpoint."""
    return _api_get_with_retry(FPL_BOOTSTRAP_URL).json()


def find_wayback_snapshot(deadline_iso: str) -> str | None:
    """Return the Wayback Machine timestamp of the closest pre-deadline bootstrap.

    Searches [deadline - WAYBACK_LOOKBACK_DAYS, deadline) for a 200-status
    snapshot. Returns None if no snapshot exists in that window.
    """
    from datetime import datetime, timezone, timedelta
    deadline_dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
    from_dt = deadline_dt - timedelta(days=WAYBACK_LOOKBACK_DAYS)
    from_str = from_dt.strftime("%Y%m%d%H%M%S")
    to_str = deadline_dt.strftime("%Y%m%d%H%M%S")

    try:
        resp = requests.get(WAYBACK_CDX_URL, params={
            "url": "fantasy.premierleague.com/api/bootstrap-static/",
            "output": "json",
            "from": from_str,
            "to": to_str,
            "filter": "statuscode:200",
            "fl": "timestamp",
            "limit": 50,
        }, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except requests.RequestException as e:
        logger.warning(f"Wayback CDX query failed: {e}")
        return None

    if len(rows) <= 1:
        return None
    return rows[-1][0]  # last = closest to deadline (CDX is chronological)


def fetch_wayback_bootstrap(timestamp: str) -> dict:
    """Fetch a specific FPL bootstrap-static snapshot from the Wayback Machine."""
    url = f"{WAYBACK_BASE}/{timestamp}/https://fantasy.premierleague.com/api/bootstrap-static/"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


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
    """Fetch all player stats for target_gw via the bulk live endpoint.

    Uses /api/event/{gw}/live/ — a single request that replaces the previous
    per-player loop (~700 calls × 0.5 s delay each ≈ 6+ minutes).

    Fields not available from the bulk endpoint (was_home, opponent_team,
    kickoff_time) are set to defaults; value/transfers/selected are
    supplemented from bootstrap_data.
    """
    url = f"{FPL_EVENT_URL}/{target_gw}/live/"
    try:
        data = _api_get_with_retry(url).json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch live GW{target_gw} data: {e}")
        return pd.DataFrame()

    team_map, element_map, pos_map = _build_bootstrap_lookups(bootstrap_data)

    # Supplement stats not present in the live endpoint from bootstrap
    bs_cost = {e["id"]: e.get("now_cost", 0) for e in bootstrap_data["elements"]}
    bs_transfers_in = {e["id"]: e.get("transfers_in_event", 0) for e in bootstrap_data["elements"]}
    bs_transfers_out = {e["id"]: e.get("transfers_out_event", 0) for e in bootstrap_data["elements"]}
    bs_selected = {e["id"]: e.get("selected_by_percent", 0) for e in bootstrap_data["elements"]}

    filter_set = set(player_ids) if player_ids else None

    rows = []
    for entry in data.get("elements", []):
        pid = entry["id"]
        if filter_set and pid not in filter_set:
            continue

        stats = entry.get("stats", {})
        element_info = element_map.get(pid, {})
        explain = entry.get("explain", [])

        row = {
            "name": element_info.get("web_name", "Unknown"),
            "position": pos_map.get(element_info.get("element_type"), "UNK"),
            "team": team_map.get(element_info.get("team"), "Unknown"),
            "element": pid,
            "GW": target_gw,
            "season": CURRENT_SEASON,
            "xP": 0.0,
            "total_points": stats.get("total_points", 0),
            "minutes": stats.get("minutes", 0),
            "goals_scored": stats.get("goals_scored", 0),
            "assists": stats.get("assists", 0),
            "clean_sheets": stats.get("clean_sheets", 0),
            "goals_conceded": stats.get("goals_conceded", 0),
            "bonus": stats.get("bonus", 0),
            "bps": stats.get("bps", 0),
            "influence": float(stats.get("influence", 0)),
            "creativity": float(stats.get("creativity", 0)),
            "threat": float(stats.get("threat", 0)),
            "ict_index": float(stats.get("ict_index", 0)),
            "value": bs_cost.get(pid, 0),
            "transfers_in": bs_transfers_in.get(pid, 0),
            "transfers_out": bs_transfers_out.get(pid, 0),
            "selected": bs_selected.get(pid, 0),
            # Not available in bulk live endpoint — set to safe defaults
            "was_home": False,
            "opponent_team": 0,
            "fixture": explain[0].get("fixture", 0) if explain else 0,
            "round": target_gw,
            "kickoff_time": "",
            "starts": stats.get("starts", 0),
            "expected_goals": float(stats.get("expected_goals", 0)),
            "expected_assists": float(stats.get("expected_assists", 0)),
            "expected_goal_involvements": float(stats.get("expected_goal_involvements", 0)),
            "expected_goals_conceded": float(stats.get("expected_goals_conceded", 0)),
        }
        for col in UNAVAILABLE_FROM_API:
            row[col] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
