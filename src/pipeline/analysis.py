"""Post-match analysis: prediction accuracy, benchmarks, and season logging."""
from __future__ import annotations
import logging
import math
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr as _spearmanr

logger = logging.getLogger(__name__)


def compute_spearman_rho(picks_df: pd.DataFrame) -> float:
    """Compute Spearman rank correlation between xP predictions and actual points.

    Returns NaN if fewer than 2 rows or no variance in either column.
    picks_df must have columns: xP, actual_points.
    """
    df = picks_df.dropna(subset=["xP", "actual_points"])
    if len(df) < 2:
        return float("nan")
    rho, _ = _spearmanr(df["xP"], df["actual_points"])
    return float(rho) if not math.isnan(rho) else float("nan")


def compute_prediction_misses(
    picks_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Compute prediction miss for each player: actual_points - xP.

    Returns list sorted by abs(miss) descending, length top_n.
    picks_df must have columns: element, name, xP, actual_points.
    """
    df = picks_df.copy()
    df["miss"] = df["actual_points"] - df["xP"]
    df = df.reindex(df["miss"].abs().sort_values(ascending=False).index)
    return df[["element", "name", "xP", "actual_points", "miss"]].head(top_n).to_dict("records")


def compute_dream_team(live_data: pd.DataFrame) -> pd.DataFrame:
    """Derive dream XI from live GW scores.

    Selects highest-scoring valid XI (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, total=11).
    Ignores club limits (FPL dream team does not apply 3-per-club rule).

    live_data must have columns: element, name, position, total_points.
    """
    from src.pipeline.optimize import select_xi
    # select_xi uses xP column — alias total_points to xP
    df = live_data.copy()
    df["xP"] = df["total_points"]
    if "now_cost" not in df.columns:
        df["now_cost"] = 50  # dummy cost
    if "team" not in df.columns:
        df["team"] = "Unknown"
    # select_xi from the full player pool as if it were the squad
    return select_xi(df)


def format_post_match_summary(
    gw: int,
    your_pts: int,
    your_xp: float,
    recommended_pts: int | None,
    recommended_xp: float | None,
    dream_pts: int | None,
    benchmarks: dict,
    your_percentile_rank: int | None,
    misses: list[dict],
) -> str:
    """Format the terminal post-match summary string."""
    lines = [
        f"\n{'='*50}",
        f"GW{gw} Post-Match Analysis",
        f"{'='*50}",
        f"Your Team:    {your_pts} pts  (predicted: {your_xp:.1f} xP)"
        + (f"  | Percentile rank: {your_percentile_rank}th" if your_percentile_rank else ""),
    ]
    if recommended_pts is not None:
        lines.append(f"Recommended:  {recommended_pts} pts  (predicted: {recommended_xp:.1f} xP)")
    if dream_pts is not None:
        lines.append(f"Dream Team:   {dream_pts} pts")

    if benchmarks:
        lines.append("\nBenchmark scores this GW:")
        for label, score in benchmarks.items():
            if score is not None:
                lines.append(f"  {label:<20}: {score} pts")

    if misses:
        lines.append("\nBiggest prediction misses (your team):")
        for m in misses:
            sign = "+" if m["miss"] >= 0 else ""
            lines.append(f"  {m['name']:<20}: predicted {m['xP']:.1f} xP, "
                         f"actual {m['actual_points']} pts  ({sign}{m['miss']:.1f})")

    if recommended_pts is not None and your_pts is not None:
        gap = recommended_pts - your_pts
        lines.append(f"\nRecommendation value: {'+' if gap >= 0 else ''}{gap} pts over your team this GW")
    if dream_pts is not None and recommended_pts is not None:
        lines.append(f"Dream team gap: {recommended_pts - dream_pts} pts (recommended vs ceiling)")

    return "\n".join(lines)


def append_accuracy_log(
    path: Path,
    gw: int,
    your_pts: int | None,
    your_xp: float | None,
    recommended_pts: int | None,
    recommended_xp: float | None,
    dream_pts: int | None = None,
    your_percentile_rank: int | None = None,
    benchmarks: dict | None = None,
    ranked_count: int | None = None,
    # B-F7 new parameters
    wildcard_pts: int | None = None,
    wildcard_xp: float | None = None,
    dream_team_pts: int | None = None,
    picks_df: pd.DataFrame | None = None,
) -> None:
    """Append one row per GW to the season accuracy log CSV.

    dream_pts and dream_team_pts are aliases; dream_team_pts takes precedence.
    picks_df: optional DataFrame with xP and actual_points columns for Spearman ρ.
    """
    from datetime import datetime, timezone
    if benchmarks is None:
        benchmarks = {}
    # Reconcile dream_pts / dream_team_pts aliases
    effective_dream_pts = dream_team_pts if dream_team_pts is not None else dream_pts

    # Compute Spearman ρ if picks_df provided
    spearman_rho = None
    if picks_df is not None and not picks_df.empty:
        spearman_rho = compute_spearman_rho(picks_df)

    row = {
        "gw": gw,
        "your_pts": your_pts,
        "your_predicted_xp": round(your_xp, 2) if your_xp is not None else None,
        "recommended_pts": recommended_pts,
        "recommended_xp": round(recommended_xp, 2) if recommended_xp is not None else None,
        "wildcard_pts": wildcard_pts,
        "wildcard_xp": round(wildcard_xp, 2) if wildcard_xp is not None else None,
        "dream_team_pts": effective_dream_pts,
        "your_percentile_rank": your_percentile_rank,
        "best_score": benchmarks.get("best_score"),
        "top_1k_score": benchmarks.get("top_1k_score"),
        "top_10k_score": benchmarks.get("top_10k_score"),
        "top_100k_score": benchmarks.get("top_100k_score"),
        "top_1m_score": benchmarks.get("top_1m_score"),
        "avg_score": benchmarks.get("avg_score"),
        "median_score": benchmarks.get("median_score"),
        "ranked_count": ranked_count,
        "spearman_rho": spearman_rho,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_existing = pd.read_csv(path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)
