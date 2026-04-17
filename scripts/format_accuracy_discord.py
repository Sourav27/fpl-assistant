"""Format the latest accuracy log row as a Discord message.

Usage:
    python scripts/format_accuracy_discord.py results/accuracy_log.csv <gw>

Prints a markdown-formatted Discord message to stdout.
"""
import argparse
import sys
import io
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def format_accuracy_row(row: pd.Series) -> str:
    gw = int(row["gw"])
    lines = [f"**GW{gw} Post-Match Accuracy**"]
    lines.append("")

    def _fmt(val, suffix="", fmt=".1f"):
        if pd.isna(val):
            return "—"
        return f"{val:{fmt}}{suffix}"

    lines.append(f"Your team:   **{_fmt(row.get('your_pts'), ' pts', 'd')}**  (predicted {_fmt(row.get('your_predicted_xp'))} xP)")
    lines.append(f"Recommended: {_fmt(row.get('recommended_pts'), ' pts', 'd')}  (predicted {_fmt(row.get('recommended_xp'))} xP)")
    lines.append(f"Optimizer squad: {_fmt(row.get('wildcard_pts'), ' pts', 'd')}  (predicted {_fmt(row.get('wildcard_xp'))} xP)")
    lines.append(f"Dream team:  {_fmt(row.get('dream_team_pts'), ' pts', 'd')}")
    lines.append("")
    lines.append(f"Prediction accuracy (Spearman ρ): {_fmt(row.get('spearman_rho'), fmt='.3f')}")
    lines.append(f"Your percentile rank: {_fmt(row.get('your_percentile_rank'), 'th', 'd')}")

    benchmarks = [
        ("Avg score",    row.get("avg_score")),
        ("Top 100k",     row.get("top_100k_score")),
        ("Top 10k",      row.get("top_10k_score")),
        ("Top 1k",       row.get("top_1k_score")),
        ("Best score",   row.get("best_score")),
    ]
    bench_lines = [f"  {lbl}: {_fmt(v, ' pts', 'd')}" for lbl, v in benchmarks if not pd.isna(v) and v is not None]
    if bench_lines:
        lines.append("")
        lines.append("**GW Benchmarks:**")
        lines.extend(bench_lines)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to accuracy_log.csv")
    parser.add_argument("gw", type=int, help="GW number to format")
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        print(f"Accuracy log not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    rows = df[df["gw"] == args.gw]
    if rows.empty:
        print(f"No row for GW{args.gw} in accuracy log", file=sys.stderr)
        sys.exit(1)

    print(format_accuracy_row(rows.iloc[-1]))


if __name__ == "__main__":
    main()
