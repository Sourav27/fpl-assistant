"""Detect whether the current GW has finished (all fixtures done).

Reads a bootstrap JSON snapshot and checks if the current GW's `finished`
flag is True.

GitHub Actions outputs written:
  - gw_finished: 'true' or 'false'
  - current_gw: the GW number (always written)
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _write_github_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def check_gw_finished(bootstrap: dict) -> tuple[bool, int | None]:
    """Return (finished, current_gw_id).

    'finished' means: current event exists AND its finished flag is True.
    """
    current = next((e for e in bootstrap["events"] if e.get("is_current")), None)
    if not current:
        return False, None
    return bool(current.get("finished", False)), current["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bootstrap_path", help="Path to bootstrap JSON snapshot")
    args = parser.parse_args()

    bootstrap = json.loads(Path(args.bootstrap_path).read_text(encoding="utf-8"))
    finished, gw = check_gw_finished(bootstrap)

    if gw is None:
        print("No current GW found in bootstrap.")
        _write_github_output("gw_finished", "false")
        sys.exit(0)

    print(f"Current GW: {gw} | Finished: {finished}")
    _write_github_output("gw_finished", str(finished).lower())
    _write_github_output("current_gw", str(gw))


if __name__ == "__main__":
    main()
