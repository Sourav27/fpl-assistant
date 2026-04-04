"""FPL user team state fetcher and post-match analysis helpers."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from src.pipeline.fetch import _api_get_with_retry
from src.config import FPL_ENTRY_URL, FPL_LEAGUES_CLASSIC_URL

logger = logging.getLogger(__name__)


@dataclass
class UserTeamState:
    """The user's current FPL team state, fetched from the public FPL API.

    All cost values are in 0.1M units (FPL convention): 77 = £7.7m.
    """
    entry_id: int
    current_squad: list[int]        # 15 element IDs (seasonal, changes each season)
    squad_codes: list[int]          # 15 persistent player codes (for cross-season joins)
    selling_prices: dict[int, int]  # element → selling price (0.1M units)
    bank: int                       # remaining budget (0.1M units, e.g. 350 = £35.0m)
    free_transfers: int             # banked free transfers, range 1-5
    active_chip: str | None         # "wildcard" | "freehit" | "bboost" | "3xc" | None
    total_value: int                # sum(selling_prices.values()) + bank

    def __post_init__(self):
        # Recompute total_value from components (ignores passed-in value)
        self.total_value = sum(self.selling_prices.values()) + self.bank
        # Cap free transfers at 5
        self.free_transfers = min(self.free_transfers, 5)


def compute_selling_price(purchase_price: int, current_price: int) -> int:
    """Compute FPL selling price in 0.1M units.

    Selling price = purchase_price + floor((current - purchase) / 2)
    FPL never charges a penalty for price drops — sell at current price if value fell.
    """
    profit = current_price - purchase_price
    if profit <= 0:
        return current_price
    return purchase_price + profit // 2


def _compute_free_transfers(gw_history: list[dict], current_gw: int) -> int:
    """Compute banked free transfers entering the NEXT gameweek.

    Logic: each unused FT banks by 1 (max 5). After using transfers, reset to 1.
    We simulate from history to find the FT count after current_gw ends.
    """
    ft = 1  # FPL starts everyone with 1 FT at GW1
    for row in sorted(gw_history, key=lambda r: r["event"]):
        if row["event"] > current_gw:
            break
        transfers_used = row.get("event_transfers", 0)
        if transfers_used == 0:
            ft = min(ft + 1, 5)  # bank 1, cap at 5
        else:
            # Using any transfers resets FTs to 1 for the next GW
            ft = 1
    return max(ft, 1)


def fetch_user_team_state(
    entry_id: int,
    gw: int,
    bootstrap_data: dict,
) -> UserTeamState:
    """Fetch user's current team state from the public FPL API.

    Makes 4 API calls: entry info, picks for current GW, transfer history, GW history.
    All cost values returned in 0.1M units.
    """
    # Build code lookup: element_id → persistent code
    code_map = {e["id"]: e["code"] for e in bootstrap_data.get("elements", [])}

    # 1. Entry info (bank balance)
    entry_data = _api_get_with_retry(f"{FPL_ENTRY_URL}/{entry_id}/").json()
    bank = entry_data.get("last_deadline_bank")
    if bank is None:
        bank = entry_data.get("bank", 0)

    # 2. Current picks
    picks_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/event/{gw}/picks/"
    ).json()
    picks = picks_data.get("picks", [])
    active_chip = picks_data.get("active_chip")

    current_squad = [p["element"] for p in picks]
    squad_codes = [code_map.get(e, e) for e in current_squad]

    # 3. Transfer history (for selling price computation)
    transfers_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/transfers/"
    ).json()
    # Build purchase price map from transfer history: element → most recent buy price
    # FPL returns transfers in reverse-chronological order (newest first)
    purchase_prices: dict[int, int] = {}
    for t in (transfers_data if isinstance(transfers_data, list) else []):
        if t["element_in"] not in purchase_prices:  # keep first (most recent) occurrence
            purchase_prices[t["element_in"]] = t["element_in_cost"]

    # Compute selling prices
    cost_map = {e["id"]: e["now_cost"] for e in bootstrap_data.get("elements", [])}
    selling_prices: dict[int, int] = {}
    for pick in picks:
        elem = pick["element"]
        # Use selling_price from picks if available (authenticated only), else compute
        if pick.get("selling_price"):
            selling_prices[elem] = pick["selling_price"]
        else:
            now = cost_map.get(elem, pick.get("purchase_price", 50))
            buy = purchase_prices.get(elem, now)  # fallback: no profit
            selling_prices[elem] = compute_selling_price(buy, now)

    # 4. GW history (for free transfer calculation)
    history_data = _api_get_with_retry(
        f"{FPL_ENTRY_URL}/{entry_id}/history/"
    ).json()
    free_transfers = _compute_free_transfers(history_data.get("current", []), gw)

    return UserTeamState(
        entry_id=entry_id,
        current_squad=current_squad,
        squad_codes=squad_codes,
        selling_prices=selling_prices,
        bank=bank,
        free_transfers=free_transfers,
        active_chip=active_chip,
        total_value=0,  # recalculated in __post_init__
    )


def fetch_gw_benchmarks(
    gw: int,
    bootstrap_data: dict,
    overall_league_id: int,
) -> dict:
    """Fetch GW benchmark scores from FPL API.

    Returns: best_score, avg_score, ranked_count, top_1k, top_10k, top_100k, top_1m (best-effort).
    best_score and avg_score come from bootstrap (already fetched). Others need standings pagination.
    """
    # Get free data from bootstrap events
    event = next((e for e in bootstrap_data.get("events", []) if e["id"] == gw), {})
    benchmarks = {
        "best_score": event.get("highest_score"),
        "avg_score": event.get("average_entry_score"),
        "ranked_count": event.get("ranked_count"),
        "top_1k_score": None,
        "top_10k_score": None,
        "top_100k_score": None,
        "top_1m_score": None,
        "median_score": None,
    }

    # Fetch score at specific ranks via standings pagination (50 entries per page)
    rank_to_key = {1000: "top_1k_score", 10000: "top_10k_score",
                   100000: "top_100k_score", 1000000: "top_1m_score"}

    for rank, key in rank_to_key.items():
        page = (rank - 1) // 50 + 1
        try:
            url = f"{FPL_LEAGUES_CLASSIC_URL}/{overall_league_id}/standings/?page_standings={page}&event={gw}"
            resp = _api_get_with_retry(url, timeout=30)
            results = resp.json().get("standings", {}).get("results", [])
            if results:
                # Score at rank position within the page
                position_in_page = (rank - 1) % 50
                if position_in_page < len(results):
                    benchmarks[key] = results[position_in_page].get("event_total") or results[position_in_page].get("total")
        except Exception as e:
            logger.warning(f"Could not fetch standings page for rank {rank}: {e}")

    return benchmarks
