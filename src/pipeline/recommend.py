"""Transfer-aware multi-GW optimizer for FPL team recommendations."""
from __future__ import annotations
import logging
from collections import defaultdict

import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as lp_value

logger = logging.getLogger(__name__)


def compute_fdr_weight(fdr: int | float, sensitivity: float) -> float:
    """Scale factor for xP based on Fixture Difficulty Rating.

    Uses fdr_team: how hard the fixture is FOR the player's team.
    FDR 1=very easy opponent → boost. FDR 5=very hard → discount.

    Formula: 1.0 - sensitivity * (fdr - 3) / 2
    Range with default sensitivity 0.15: [0.85, 1.15]
    """
    weight = 1.0 - sensitivity * (fdr - 3) / 2
    return max(0.0, weight)  # clamp to non-negative


def build_fixture_fdr_map(
    fixtures: list[dict],
    gws: list[int],
) -> dict[tuple[int, int], float]:
    """Build {(team_id, gw): fdr_team} mapping from fixtures list.

    fdr_team = team_h_difficulty if player's team is home,
               team_a_difficulty if player's team is away.

    Double-GW teams get the average FDR across their fixtures.
    Teams with no fixture in a GW are absent from the map (blank GW → xP = 0).
    """
    gw_set = set(gws)
    # Accumulate FDR values per (team, gw) — handle double GWs
    fdr_accumulator: dict[tuple[int, int], list[float]] = defaultdict(list)

    for f in fixtures:
        gw = f.get("event")
        if gw not in gw_set:
            continue
        team_h = f["team_h"]
        team_a = f["team_a"]
        fdr_h = f.get("team_h_difficulty", 3)
        fdr_a = f.get("team_a_difficulty", 3)
        fdr_accumulator[(team_h, gw)].append(fdr_h)
        fdr_accumulator[(team_a, gw)].append(fdr_a)

    return {key: sum(vals) / len(vals) for key, vals in fdr_accumulator.items()}


def recommend_transfers(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    horizon: int,
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Compute optimal transfer plan for 1 to 5 GWs ahead.

    Args:
        user_state: Current squad, bank, free transfers from FPL API.
        predictions: Full player predictions CSV (element, code, name, position, team, xP, now_cost).
                     now_cost is in 0.1M units (FPL convention: 105 = £10.5m). Do NOT divide by 10 here.
        fixtures: All FPL fixtures from fetch_fixtures().
        horizon: Number of GWs to plan (1=single GW, 2-5=multi-GW ILP).
        fdr_sensitivity: FDR weight sensitivity (0=ignore, 0.15=default).
        max_hit_points: Max penalty per GW. E.g. 8 = max 2 extra transfers.

    Returns dict with keys: transfers (list), projected_xp (float), hit_cost (int),
        bank_after (float), squad_after (list of element IDs).
    """
    if horizon == 1:
        return _recommend_single_gw(
            user_state, predictions, fixtures, fdr_sensitivity, max_hit_points
        )
    return _recommend_multi_gw(
        user_state, predictions, fixtures, horizon, fdr_sensitivity, max_hit_points
    )


def _recommend_single_gw(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Single-GW optimiser: find best transfers respecting budget and hit cap."""
    from src.config import SQUAD_RULES

    # Build player pool: current squad + all available players
    current_squad = set(user_state.current_squad)

    # All values in 0.1M units (FPL convention) — no division needed
    # user_state.selling_prices: element → 0.1M units
    # user_state.bank: 0.1M units
    # predictions.now_cost: 0.1M units (from save_full_predictions)
    selling_prices_01m = user_state.selling_prices
    bank_01m = user_state.bank
    # bank_pounds for display in output (convert 0.1M units to £M)
    bank_pounds = user_state.bank / 10

    ft = user_state.free_transfers
    max_hits = max_hit_points // 4  # number of extra transfers allowed

    # Use ILP to find optimal single-GW squad
    n = len(predictions)
    players = predictions.reset_index(drop=True)

    prob = LpProblem("FPL_SingleGW_Recommend", LpMaximize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    transfer_in = [LpVariable(f"tin_{i}", cat="Binary") for i in range(n)]
    transfer_out = [LpVariable(f"tout_{i}", cat="Binary") for i in range(n)]
    captain = [LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    hits = LpVariable("hits", lowBound=0, cat="Integer")

    in_squad = [1 if players.iloc[i]["element"] in current_squad else 0 for i in range(n)]

    # Objective: total xP + captain bonus (captain doubles xP, so add xP once more) - hit cost
    xp = [float(players.iloc[i]["xP"]) for i in range(n)]
    prob += (
        lpSum(x[i] * xp[i] for i in range(n))
        + lpSum(captain[i] * xp[i] for i in range(n))
        - 4 * hits
    )

    # Squad size = 15
    prob += lpSum(x) == 15

    # Transfer continuity: x = in_squad + transfer_in - transfer_out
    for i in range(n):
        prob += x[i] == in_squad[i] + transfer_in[i] - transfer_out[i]
        prob += transfer_in[i] + transfer_out[i] <= 1  # can't both in and out same player

    # Transfer count
    transfers_used = lpSum(transfer_in)
    prob += hits >= transfers_used - ft
    prob += hits <= max_hits
    prob += hits >= 0

    # Budget: bank + sales revenue >= purchase cost (all in 0.1M units)
    prob += (
        bank_01m
        + lpSum(transfer_out[i] * selling_prices_01m.get(players.iloc[i]["element"], players.iloc[i]["now_cost"])
                for i in range(n))
        >= lpSum(transfer_in[i] * players.iloc[i]["now_cost"] for i in range(n))
    )

    # Position constraints
    for pos, count in SQUAD_RULES["positions"].items():
        mask = [1 if players.iloc[i]["position"] == pos else 0 for i in range(n)]
        prob += lpSum(x[i] for i in range(n) if mask[i]) == count

    # Max 3 per team
    for team in players["team"].unique():
        mask = (players["team"] == team).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) <= SQUAD_RULES["max_per_team"]

    # Captain: exactly 1
    prob += lpSum(captain) == 1
    # Captain must be in XI (simplification: captain must be in squad)
    for i in range(n):
        prob += captain[i] <= x[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n) if lp_value(x[i]) is not None and lp_value(x[i]) > 0.5]
    ins = [i for i in range(n) if lp_value(transfer_in[i]) is not None and lp_value(transfer_in[i]) > 0.5]
    outs = [i for i in range(n) if lp_value(transfer_out[i]) is not None and lp_value(transfer_out[i]) > 0.5]
    hit_count = int(round(lp_value(hits) or 0))
    cap_idx = next((i for i in range(n) if lp_value(captain[i]) is not None and lp_value(captain[i]) > 0.5), None)

    transfers = []
    for out_i, in_i in zip(outs, ins):
        p_out = players.iloc[out_i]
        p_in = players.iloc[in_i]
        transfers.append({
            "player_out": p_out["name"],
            "player_in": p_in["name"],
            "price_out": selling_prices_01m.get(p_out["element"], p_out["now_cost"]) / 10,  # convert to £ for display
            "price_in": p_in["now_cost"] / 10,  # convert to £ for display
            "xp_out": p_out["xP"],
            "xp_in": p_in["xP"],
        })

    squad_after = [players.iloc[i]["element"] for i in selected]
    projected_xp = sum(players.iloc[i]["xP"] * (1 + (1 if i == cap_idx else 0))
                       for i in selected)

    return {
        "transfers": transfers,
        "projected_xp": projected_xp,
        "hit_cost": hit_count * 4,
        "bank_after": bank_pounds,
        "squad_after": squad_after,
    }
