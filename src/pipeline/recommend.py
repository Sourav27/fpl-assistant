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
