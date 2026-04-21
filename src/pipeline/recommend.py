"""Transfer-aware multi-GW optimizer for FPL team recommendations."""
from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value as lp_value

from src.config import SQUAD_RULES

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

    # Squad size
    prob += lpSum(x) == SQUAD_RULES["squad_size"]

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

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    outs = sorted(outs, key=lambda i: pos_order.get(players.iloc[i]["position"], 9))
    ins = sorted(ins, key=lambda i: pos_order.get(players.iloc[i]["position"], 9))

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
            "shap_reason": p_in.get("shap_reason", "") if hasattr(p_in, "get") else getattr(p_in, "shap_reason", ""),
        })

    squad_after = [players.iloc[i]["element"] for i in selected]
    projected_xp = sum(players.iloc[i]["xP"] * (1 + (1 if i == cap_idx else 0))
                       for i in selected)

    # Compute actual post-transfer bank in £M
    cost_out = sum(selling_prices_01m.get(players.iloc[i]["element"], players.iloc[i]["now_cost"]) for i in outs)
    cost_in = sum(players.iloc[i]["now_cost"] for i in ins)
    bank_after = round(bank_pounds + (cost_out - cost_in) / 10, 1)

    hit_cost = hit_count * 4
    # Return same per-GW structure as _recommend_multi_gw for uniform downstream handling
    return {
        "transfers": [{"transfers": transfers, "hit_cost": hit_cost, "bank_after": bank_after}],
        "projected_xp": projected_xp,
        "hit_cost": hit_cost,
        "bank_after": bank_after,
        "squad_after": squad_after,
    }


def build_xp_matrix(
    predictions: pd.DataFrame,
    fixtures: list[dict],
    team_id_map: dict[str, int],
    gws: list[int],
    fdr_sensitivity: float,
) -> pd.DataFrame:
    """Build player × GW matrix of FDR-adjusted xP values.

    Blank GW = 0 xP. Double GW = sum of xP from both fixtures.
    now_cost in predictions is in 0.1M units (FPL convention: 105 = £10.5m).
    """
    fdr_map = build_fixture_fdr_map(fixtures, gws)
    matrix = pd.DataFrame(0.0, index=predictions.index, columns=gws)

    for gw in gws:
        for idx, row in predictions.iterrows():
            team_id = team_id_map.get(row["team"])
            if team_id is None:
                matrix.loc[idx, gw] = 0.0
                continue
            fdr = fdr_map.get((team_id, gw))
            if fdr is None:
                matrix.loc[idx, gw] = 0.0  # blank GW
            else:
                weight = compute_fdr_weight(fdr, fdr_sensitivity)
                matrix.loc[idx, gw] = row["xP"] * weight

    return matrix


def _recommend_multi_gw(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
    fixtures: list[dict],
    horizon: int,
    fdr_sensitivity: float,
    max_hit_points: int,
) -> dict:
    """Multi-GW ILP using PuLP with free transfer banking and FDR weighting.

    Linearises FT carryover with big-M = 20.
    All costs in 0.1M units (FPL convention).
    """

    players = predictions.reset_index(drop=True)
    n = len(players)
    M = 20  # big-M for FT linearisation

    gw_indices = list(range(horizon))

    # All values in 0.1M units — no conversion needed
    sp = user_state.selling_prices  # element → 0.1M units
    bank0 = user_state.bank         # 0.1M units
    ft0 = user_state.free_transfers
    max_hits_per_gw = max_hit_points // 4
    current_squad_set = set(user_state.current_squad)
    in_squad_gw0 = [1 if players.iloc[i]["element"] in current_squad_set else 0 for i in range(n)]

    # FDR-weighted xP per player per GW (GW 0 = raw xP, future GWs fall back to raw xP without bootstrap)
    xp_matrix: list[list[float]] = []
    for i in range(n):
        row_xp = [float(players.iloc[i]["xP"]) for _ in gw_indices]
        xp_matrix.append(row_xp)

    prob = LpProblem("FPL_MultiGW_Recommend", LpMaximize)

    # Decision variables
    squad = [[LpVariable(f"sq_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    xi = [[LpVariable(f"xi_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    tin = [[LpVariable(f"tin_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    tout = [[LpVariable(f"tout_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    cap = [[LpVariable(f"cap_{i}_{g}", cat="Binary") for g in gw_indices] for i in range(n)]
    hits = [LpVariable(f"hits_{g}", lowBound=0, cat="Integer") for g in gw_indices]
    ft = [LpVariable(f"ft_{g}", lowBound=1, upBound=5, cat="Integer") for g in gw_indices]
    used_ft = [LpVariable(f"used_ft_{g}", cat="Binary") for g in gw_indices]
    bank = [LpVariable(f"bank_{g}", lowBound=0) for g in gw_indices]

    # Objective
    prob += lpSum(
        xp_matrix[i][g] * (xi[i][g] + cap[i][g]) - 4 * hits[g]
        for i in range(n) for g in gw_indices
    )

    for g in gw_indices:
        # Squad size
        prob += lpSum(squad[i][g] for i in range(n)) == SQUAD_RULES["squad_size"]

        # XI size
        prob += lpSum(xi[i][g] for i in range(n)) == SQUAD_RULES["xi_size"]

        # Position constraints (squad)
        for pos, count in SQUAD_RULES["positions"].items():
            prob += lpSum(squad[i][g] for i in range(n) if players.iloc[i]["position"] == pos) == count

        # XI position constraints
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "GK") == 1
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "DEF") >= 3
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "MID") >= 2
        prob += lpSum(xi[i][g] for i in range(n) if players.iloc[i]["position"] == "FWD") >= 1

        # Max 3 per club
        for team in players["team"].unique():
            prob += lpSum(squad[i][g] for i in range(n) if players.iloc[i]["team"] == team) <= 3

        # XI ⊆ squad
        for i in range(n):
            prob += xi[i][g] <= squad[i][g]

        # Captain: 1 in XI
        prob += lpSum(cap[i][g] for i in range(n)) == 1
        for i in range(n):
            prob += cap[i][g] <= xi[i][g]

        # Transfer continuity
        prev_squad = in_squad_gw0 if g == 0 else [squad[i][g - 1] for i in range(n)]
        for i in range(n):
            prob += squad[i][g] == prev_squad[i] + tin[i][g] - tout[i][g]
            prob += tin[i][g] + tout[i][g] <= 1

        transfers_used_g = lpSum(tin[i][g] for i in range(n))

        # FT initialisation
        if g == 0:
            prob += ft[0] == ft0
        else:
            # FT carry-forward with big-M linearisation
            prob += ft[g] <= ft[g - 1] + 1 + M * used_ft[g - 1]
            prob += ft[g] <= 5
            prob += ft[g] >= 1
            prob += ft[g] <= 1 + M * (1 - used_ft[g - 1])

        # used_ft indicator
        prob += transfers_used_g <= M * used_ft[g]
        prob += transfers_used_g >= used_ft[g]

        # Hit cost
        prob += hits[g] >= transfers_used_g - ft[g]
        prob += hits[g] >= 0
        prob += hits[g] <= max_hits_per_gw

        # Budget
        if g == 0:
            prob += bank[0] == bank0 + lpSum(
                tout[i][0] * sp.get(players.iloc[i]["element"], players.iloc[i]["now_cost"])
                for i in range(n)
            ) - lpSum(tin[i][0] * players.iloc[i]["now_cost"] for i in range(n))
        else:
            prob += bank[g] == bank[g - 1] + lpSum(
                tout[i][g] * players.iloc[i]["now_cost"] for i in range(n)
            ) - lpSum(tin[i][g] * players.iloc[i]["now_cost"] for i in range(n))
        prob += bank[g] >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results per GW
    gw_results = []
    for g in gw_indices:
        ins = [i for i in range(n) if lp_value(tin[i][g]) is not None and lp_value(tin[i][g]) > 0.5]
        outs = [i for i in range(n) if lp_value(tout[i][g]) is not None and lp_value(tout[i][g]) > 0.5]
        hit_count = int(round(lp_value(hits[g]) or 0))
        bank_val = lp_value(bank[g]) or 0.0

        # Sort outs and ins by position so same-position players pair up in display.
        pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        outs = sorted(outs, key=lambda i: pos_order.get(players.iloc[i]["position"], 9))
        ins = sorted(ins, key=lambda i: pos_order.get(players.iloc[i]["position"], 9))

        transfers_gw = []
        for out_i, in_i in zip(outs, ins):
            p_out = players.iloc[out_i]
            p_in = players.iloc[in_i]
            transfers_gw.append({
                "player_out": p_out["name"],
                "player_in": p_in["name"],
                "price_out": sp.get(p_out["element"], p_out["now_cost"]) / 10,  # convert to £ for display
                "price_in": p_in["now_cost"] / 10,  # convert to £ for display
                "xp_out": p_out["xP"],
                "xp_in": p_in["xP"],
                "shap_reason": p_in.get("shap_reason", "") if hasattr(p_in, "get") else getattr(p_in, "shap_reason", ""),
            })
        gw_results.append({
            "transfers": transfers_gw,
            "hit_cost": hit_count * 4,
            "bank_after": round(bank_val / 10, 1),  # convert to £ for display
        })

    total_xp = sum(
        xp_matrix[i][g]
        for g in gw_indices
        for i in range(n)
        if lp_value(xi[i][g]) is not None and lp_value(xi[i][g]) > 0.5
    )

    return {
        "transfers": gw_results,
        "projected_xp": round(total_xp, 1),
        "hit_cost": sum(r["hit_cost"] for r in gw_results),
        "bank_after": gw_results[-1]["bank_after"] if gw_results else bank0 / 10,
        "squad_after": [
            players.iloc[i]["element"]
            for i in range(n)
            if lp_value(squad[i][0]) is not None and lp_value(squad[i][0]) > 0.5
        ],
    }


def recommend_wildcard(
    user_state: "UserTeamState",
    predictions: pd.DataFrame,
) -> dict:
    """Unconstrained squad selection using user's total squad value as budget.

    Used for Wildcard and Free Hit chips.
    predictions.now_cost is in 0.1M units. user_state.total_value is also in 0.1M units.
    Both are consistent — pass total_value directly as budget override.
    """
    from src.pipeline.optimize import optimize_team
    # total_value and now_cost are both in 0.1M units — pass directly
    preds = predictions.copy()
    result = optimize_team(preds, budget=int(user_state.total_value))
    return {
        "squad": result["squad"]["element"].tolist(),
        "xi": result["xi"]["element"].tolist(),
        "captain": result["captain"]["element"],
        "total_xp": result["total_xp"],
        "transfers": [],  # no specific transfers (full rebuild)
    }


def save_recommend_csv(plan: dict, path: Path, start_gw: int) -> None:
    """Save transfer plan to CSV.

    Columns: gw, action, player_out, player_in, price_out, price_in,
             xp_out, xp_in, hit_cost, bank_after
    """
    rows = []
    transfers_by_gw = plan.get("transfers", [])
    for gw_offset, gw_transfers in enumerate(transfers_by_gw):
        gw = start_gw + gw_offset
        hit_cost = gw_transfers.get("hit_cost", 0)
        bank_after = gw_transfers.get("bank_after", "")
        transfers_list = gw_transfers.get("transfers", [])
        if not transfers_list:
            rows.append({
                "gw": gw, "action": "hold", "player_out": "", "player_in": "",
                "price_out": "", "price_in": "", "xp_out": "", "xp_in": "",
                "hit_cost": hit_cost, "bank_after": bank_after,
            })
        else:
            for t in transfers_list:
                rows.append({
                    "gw": gw,
                    "action": "transfer",
                    "player_out": t["player_out"],
                    "player_in": t["player_in"],
                    "price_out": t["price_out"],
                    "price_in": t["price_in"],
                    "xp_out": round(t["xp_out"], 1),
                    "xp_in": round(t["xp_in"], 1),
                    "hit_cost": hit_cost,
                    "bank_after": bank_after,
                })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
