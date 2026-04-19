"""Generate FPL performance reports as PNG charts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
REPORTS_DIR = RESULTS_DIR / "reports"


def load_accuracy_log(path: Path, from_gw: int = 31) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "season" not in df.columns:
        df["season"] = "2025-26"
    df = df[df["gw"] >= from_gw].sort_values(["season", "gw"]).reset_index(drop=True)
    return df


def estimate_rank_percentile(score: float, best_score: float, avg_score: float) -> float:
    """Linear interpolation: best_score→0.001%, avg_score→50%, 0→100%."""
    anchors_pts = [0, avg_score, best_score]
    anchors_pct = [100.0, 50.0, 0.001]
    pct = float(np.interp(score, anchors_pts, anchors_pct))
    return max(0.001, min(100.0, pct))


def _decision_impact(accuracy_df: pd.DataFrame, season: str) -> dict[int, float]:
    transfers_path = RESULTS_DIR / season / "actual_transfers.csv"
    if not transfers_path.exists():
        return {}
    transfers = pd.read_csv(transfers_path)
    impact = {}
    for gw, _ in accuracy_df.groupby("gw"):
        rec_path = RESULTS_DIR / season / f"gw{gw}" / "recommend.csv"
        rec_gain = 0.0
        if rec_path.exists():
            rec = pd.read_csv(rec_path)
            rec_gw = rec[rec["gw"] == gw]
            rec_gain = float((rec_gw["xp_in"] - rec_gw["xp_out"]).sum()) if not rec_gw.empty else 0.0
        gw_transfers = transfers[transfers["gw"] == gw]
        actual_gain = float(gw_transfers["actual_pts_gained"].sum()) if not gw_transfers.empty else 0.0
        impact[int(gw)] = actual_gain - rec_gain
    return impact


def plot_gw_chart(accuracy_df: pd.DataFrame, out_path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_bars, ax_impact) = plt.subplots(
        2, 1, figsize=(max(10, len(accuracy_df) * 1.2), 10),
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    x = np.arange(len(accuracy_df))
    width = 0.25
    for i, (col, label, colour) in enumerate([
        ("your_pts",        "My team",      "#4472C4"),
        ("wildcard_pts",    "Optimal",      "#70AD47"),
        ("recommended_pts", "Recommended",  "#ED7D31"),
    ]):
        vals = accuracy_df[col].fillna(0).tolist()
        bars = ax_bars.bar(x + (i - 1) * width, vals, width, label=label, color=colour)
        for bar, row, val in zip(bars, accuracy_df.itertuples(), vals):
            if val == 0:
                continue
            best = getattr(row, "best_score", None)
            avg = getattr(row, "avg_score", None)
            if col == "your_pts" and pd.notna(getattr(row, "your_percentile_rank", None)):
                pct_label = f"top{row.your_percentile_rank:.0f}%"
            elif best and avg:
                pct = estimate_rank_percentile(val, best, avg)
                pct_label = f"top{pct:.1f}%" if pct >= 0.1 else f"top{pct:.3f}%"
            else:
                pct_label = ""
            ax_bars.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{int(val)}pts\n({pct_label})" if pct_label else f"{int(val)}pts",
                         ha="center", va="bottom", fontsize=7)

    seasons = accuracy_df["season"].tolist()
    for idx in range(1, len(seasons)):
        if seasons[idx] != seasons[idx - 1]:
            ax_bars.axvline(x=idx - 0.5, color="gray", linestyle="--", linewidth=0.8)

    tick_labels = [f"GW{r.gw}" for r in accuracy_df.itertuples()]
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_bars.set_ylabel("Points scored")
    ax_bars.set_title("GW Performance: My Team vs Optimal vs Recommended")
    ax_bars.legend()

    impact = {}
    for s in accuracy_df["season"].unique():
        impact.update(_decision_impact(accuracy_df[accuracy_df["season"] == s], s))
    impact_vals = [impact.get(gw, 0.0) for gw in accuracy_df["gw"].tolist()]
    bar_colours = ["#70AD47" if v >= 0 else "#FF0000" for v in impact_vals]
    ax_impact.bar(x, impact_vals, color=bar_colours)
    ax_impact.axhline(0, color="black", linewidth=0.8)
    ax_impact.set_xticks(x)
    ax_impact.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_impact.set_ylabel("Transfer impact (pts)")
    ax_impact.set_title("Decision Impact vs Recommendation")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reports] Saved {out_path}")


def plot_season_chart(accuracy_df: pd.DataFrame, out_path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(accuracy_df) * 0.8), 6), constrained_layout=True)
    for col, label, colour in [
        ("your_pts",        "My team",      "#4472C4"),
        ("wildcard_pts",    "Optimal",      "#70AD47"),
        ("recommended_pts", "Recommended",  "#ED7D31"),
    ]:
        cumulative = accuracy_df[col].fillna(0).cumsum()
        ax.plot(range(len(accuracy_df)), cumulative, marker="o", label=label, color=colour)
    tick_labels = [f"GW{r.gw}" for r in accuracy_df.itertuples()]
    ax.set_xticks(range(len(accuracy_df)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative points")
    ax.set_title("Cumulative Season Performance")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reports] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-gw", type=int, default=31)
    parser.add_argument("--accuracy-log", type=Path, default=RESULTS_DIR / "accuracy_log.csv")
    args = parser.parse_args()

    if not args.accuracy_log.exists():
        print(f"[reports] {args.accuracy_log} not found — nothing to plot")
        return

    df = load_accuracy_log(args.accuracy_log, from_gw=args.from_gw)
    if df.empty:
        print(f"[reports] No data from GW{args.from_gw} onwards")
        return

    plot_gw_chart(df, REPORTS_DIR / "rank_comparison_gw.png")
    plot_season_chart(df, REPORTS_DIR / "rank_comparison_season.png")


if __name__ == "__main__":
    main()
