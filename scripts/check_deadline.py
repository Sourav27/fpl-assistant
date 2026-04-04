"""Check whether the next FPL GW deadline is within 48 hours.

Reads a bootstrap JSON file, computes hours until the next GW deadline,
and writes GitHub Actions output variables:
  - deadline_approaching: 'true' or 'false'
  - next_gw: the GW number (only when approaching)

Usage:
  python scripts/check_deadline.py results/snapshots/bootstrap_gw32.json
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def hours_until_deadline(bootstrap: dict) -> tuple[float | None, int | None]:
    """Return (hours_until_deadline, next_gw_id) or (None, None) if no next GW."""
    next_event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if not next_event:
        return None, None
    deadline_str = next_event["deadline_time"].replace("Z", "+00:00")
    deadline = datetime.fromisoformat(deadline_str)
    now = datetime.now(timezone.utc)
    hours = (deadline - now).total_seconds() / 3600
    return hours, next_event["id"]


def _write_github_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bootstrap_path", help="Path to bootstrap JSON snapshot file")
    args = parser.parse_args()

    bootstrap = json.loads(Path(args.bootstrap_path).read_text(encoding="utf-8"))
    hours, next_gw = hours_until_deadline(bootstrap)

    if hours is None:
        print("No upcoming GW found in bootstrap — skipping deadline check.")
        _write_github_output("deadline_approaching", "false")
        sys.exit(0)

    approaching = hours < 48.0
    print(f"Next GW: GW{next_gw} | Hours until deadline: {hours:.1f} | Approaching: {approaching}")
    _write_github_output("deadline_approaching", str(approaching).lower())
    _write_github_output("hours_until", f"{hours:.1f}")
    if next_gw is not None:
        _write_github_output("next_gw", str(next_gw))


if __name__ == "__main__":
    main()
