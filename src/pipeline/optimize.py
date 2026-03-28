# src/pipeline/optimize.py
"""PuLP-based FPL team optimizer — replaces R lpSolve scripts."""
import pandas as pd
import pulp
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from src.config import SQUAD_RULES


def select_squad(players: pd.DataFrame, budget: int | None = None) -> pd.DataFrame:
    """Select optimal 15-player squad using linear programming."""
    budget = budget if budget is not None else SQUAD_RULES["budget"]
    prob = LpProblem("FPL_Squad", LpMaximize)
    n = len(players)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize total xP
    prob += lpSum(x[i] * players.iloc[i]["xP"] for i in range(n))

    # Squad size = 15
    prob += lpSum(x) == SQUAD_RULES["squad_size"]

    # Budget constraint
    prob += lpSum(x[i] * players.iloc[i]["now_cost"] for i in range(n)) <= budget

    # Position constraints
    for pos, count in SQUAD_RULES["positions"].items():
        mask = (players["position"] == pos).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) == count

    # Max 3 per team
    for team in players["team"].unique():
        mask = (players["team"] == team).values
        prob += lpSum(x[i] for i in range(n) if mask[i]) <= SQUAD_RULES["max_per_team"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n) if value(x[i]) is not None and value(x[i]) > 0.5]
    return players.iloc[selected].reset_index(drop=True)


def select_xi(squad: pd.DataFrame) -> pd.DataFrame:
    """Select best 11 from a 15-player squad."""
    prob = LpProblem("FPL_XI", LpMaximize)
    n = len(squad)
    x = [LpVariable(f"xi_{i}", cat="Binary") for i in range(n)]

    prob += lpSum(x[i] * squad.iloc[i]["xP"] for i in range(n))

    # Exactly 11
    prob += lpSum(x) == SQUAD_RULES["xi_size"]

    # Exactly 1 GK
    gk_mask = (squad["position"] == "GK").values
    prob += lpSum(x[i] for i in range(n) if gk_mask[i]) == 1

    # DEF: 3-5
    def_mask = (squad["position"] == "DEF").values
    prob += lpSum(x[i] for i in range(n) if def_mask[i]) >= 3
    prob += lpSum(x[i] for i in range(n) if def_mask[i]) <= 5

    # MID: 2-5
    mid_mask = (squad["position"] == "MID").values
    prob += lpSum(x[i] for i in range(n) if mid_mask[i]) >= 2
    prob += lpSum(x[i] for i in range(n) if mid_mask[i]) <= 5

    # FWD: 1-3
    fwd_mask = (squad["position"] == "FWD").values
    prob += lpSum(x[i] for i in range(n) if fwd_mask[i]) >= 1
    prob += lpSum(x[i] for i in range(n) if fwd_mask[i]) <= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected = [i for i in range(n) if value(x[i]) is not None and value(x[i]) > 0.5]
    return squad.iloc[selected].reset_index(drop=True)


def select_captain(xi: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Pick captain (highest xP) and vice-captain (second highest)."""
    sorted_xi = xi.sort_values("xP", ascending=False)
    return sorted_xi.iloc[0], sorted_xi.iloc[1]


def optimize_team(players: pd.DataFrame, budget: int | None = None) -> dict:
    """Full optimization pipeline: squad -> XI -> captain."""
    squad = select_squad(players, budget=budget)
    xi = select_xi(squad)
    captain, vice = select_captain(xi)

    return {
        "squad": squad,
        "xi": xi,
        "captain": captain,
        "vice_captain": vice,
        "total_xp": xi["xP"].sum() + captain["xP"],  # captain points doubled
        "bench": squad[~squad["element"].isin(xi["element"])],
    }
