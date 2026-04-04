"""Format predict/recommend CSV outputs into Discord-ready messages.

Two blocks:
  format_wildcard_xi_block — optimal unconstrained XI (xi_gw{N}.csv)
  format_my_team_block     — post-transfer 15-man squad with bench

Usage:
  python scripts/format_discord_results.py \\
      results/xi_gw32.csv \\
      results/squad_recommend_gw32.csv \\
      results/xi_recommend_gw32.csv \\
      results/recommend_gw32.csv \\
      32
"""
import argparse
import csv
import sys
from pathlib import Path


def format_wildcard_xi_block(rows: list[dict], gw: int) -> str:
    """Optimal unconstrained Starting XI, grouped by position, captain marked."""
    if not rows:
        return f"GW{gw} Wildcard XI: no data."

    captain = max(rows, key=lambda r: float(r["xP"]))
    lines = [f"**Wildcard XI — GW{gw}**"]
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_rows = sorted(
            [r for r in rows if r["position"] == pos],
            key=lambda r: float(r["xP"]), reverse=True,
        )
        if not pos_rows:
            continue
        lines.append(f"\n{pos}")
        for r in pos_rows:
            cap = " (C)" if str(r["element"]) == str(captain["element"]) else ""
            lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP{cap}")
    return "\n".join(lines)


def format_my_team_block(
    squad_rows: list[dict],
    xi_rows: list[dict],
    rec_rows: list[dict],
    bank: float,
    gw: int,
    free_transfers: int | None = None,
) -> str:
    """Post-transfer 15-man squad: starters first by position, then bench."""
    if not squad_rows:
        return f"GW{gw} My Team: no data."

    xi_elements = {str(r["element"]) for r in xi_rows}
    captain = max(xi_rows, key=lambda r: float(r["xP"])) if xi_rows else None

    starters = [r for r in squad_rows if str(r["element"]) in xi_elements]
    bench = [r for r in squad_rows if str(r["element"]) not in xi_elements]

    ft_str = f"transfers left: {free_transfers}, " if free_transfers is not None else ""
    lines = [f"**My Team After Transfers — GW{gw}** ({ft_str}bank: £{bank:.1f}m)"]

    # Transfers summary (current GW only)
    gw_transfers = [r for r in rec_rows if int(r["gw"]) == gw and r.get("action") == "transfer"]
    if gw_transfers:
        lines.append("\nTransfers")
        for t in gw_transfers:
            hit = f" (-{t['hit_cost']}pts)" if float(t["hit_cost"]) > 0 else ""
            lines.append(
                f"- {t['player_out']} → {t['player_in']}"
                f"  £{float(t['price_out']):.1f}→£{float(t['price_in']):.1f}{hit}"
            )
    else:
        lines.append("\nTransfers: hold")

    # Starting XI grouped by position
    lines.append("\nStarting XI")
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_rows = sorted(
            [r for r in starters if r["position"] == pos],
            key=lambda r: float(r["xP"]), reverse=True,
        )
        for r in pos_rows:
            cap = " (C)" if captain and str(r["element"]) == str(captain["element"]) else ""
            lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP{cap}")

    # Bench
    bench_sorted = sorted(bench, key=lambda r: float(r["xP"]), reverse=True)
    lines.append("\nBench")
    for r in bench_sorted:
        lines.append(f"• {r['name']} ({r['team']}) — {float(r['xP']):.1f} xP")

    return "\n".join(lines)


def _read_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xi_path",        help="results/xi_gw{N}.csv")
    parser.add_argument("squad_rec_path", help="results/squad_recommend_gw{N}.csv")
    parser.add_argument("xi_rec_path",    help="results/xi_recommend_gw{N}.csv")
    parser.add_argument("recommend_path", help="results/recommend_gw{N}.csv")
    parser.add_argument("gw", type=int,   help="Target GW number")
    args = parser.parse_args()

    xi_rows   = _read_csv(args.xi_path)
    squad_rec = _read_csv(args.squad_rec_path)
    xi_rec    = _read_csv(args.xi_rec_path)
    rec_rows  = _read_csv(args.recommend_path)

    gw_rec = [r for r in rec_rows if int(r["gw"]) == args.gw]
    bank = float(gw_rec[-1]["bank_after"]) if gw_rec else 0.0
    free_transfers = int(squad_rec[0]["free_transfers_after"]) if squad_rec and "free_transfers_after" in squad_rec[0] else None

    print(format_wildcard_xi_block(xi_rows, args.gw))
    print()
    print(format_my_team_block(squad_rec, xi_rec, rec_rows, bank, args.gw, free_transfers))


if __name__ == "__main__":
    main()
