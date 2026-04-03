"""Fetch and cache FPL bootstrap-static snapshots for every GW.

Two modes:

  Historical backfill (default):
    Fetches pre-deadline bootstrap for every past GW using the Wayback Machine
    CDX API. Targets the snapshot closest to (but before) each GW deadline.
    Skips GWs that already have a cached snapshot.

      python scripts/fetch_bootstrap_snapshots.py

  Single GW (--gw N):
    Backfills one specific GW only.

      python scripts/fetch_bootstrap_snapshots.py --gw 30

  Live (--live):
    Fetches the current bootstrap from the FPL API and saves it as the
    snapshot for the current GW. Used by the daily GitHub Actions cron job.

      python scripts/fetch_bootstrap_snapshots.py --live

Output: results/snapshots/bootstrap_gw{N}.json
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "results" / "snapshots"

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"

# How many days before deadline to start looking for a Wayback snapshot.
# FPL squads are stable in this window — transfer deadline hasn't passed.
LOOKBACK_DAYS = 7

# Polite delay between Wayback Machine requests (seconds).
WAYBACK_DELAY = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_live_bootstrap() -> dict:
    print(f"  Fetching live bootstrap from {FPL_BOOTSTRAP_URL}")
    return _get(FPL_BOOTSTRAP_URL).json()


def find_wayback_snapshot(deadline_iso: str) -> str | None:
    """Return the best Wayback Machine timestamp for the pre-deadline FPL bootstrap.

    Searches [deadline - LOOKBACK_DAYS, deadline) for a 200-status snapshot,
    preferring the one closest to (but before) the deadline.
    """
    deadline_dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
    from_dt = deadline_dt.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # Move back LOOKBACK_DAYS days
    from_ts = int((from_dt.timestamp() - LOOKBACK_DAYS * 86400))
    from_str = datetime.fromtimestamp(from_ts, tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    to_str = deadline_dt.strftime("%Y%m%d%H%M%S")

    params = {
        "url": "fantasy.premierleague.com/api/bootstrap-static/",
        "output": "json",
        "from": from_str,
        "to": to_str,
        "filter": "statuscode:200",
        "fl": "timestamp",
        "limit": 50,
    }
    resp = _get(WAYBACK_CDX_URL, params=params)
    rows = resp.json()

    # rows[0] is the header ["timestamp"]; remaining rows are results
    if len(rows) <= 1:
        return None

    # Last entry is closest to deadline (CDX returns chronological order)
    return rows[-1][0]


def fetch_wayback_bootstrap(timestamp: str) -> dict:
    url = f"{WAYBACK_BASE}/{timestamp}/https://fantasy.premierleague.com/api/bootstrap-static/"
    print(f"  Fetching Wayback snapshot {timestamp} ...")
    return _get(url, timeout=60).json()


def save_snapshot(bootstrap: dict, gw: int) -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"bootstrap_gw{gw}.json"
    path.write_text(json.dumps(bootstrap), encoding="utf-8")
    return path


def gw_events(bootstrap: dict) -> list[dict]:
    """Return all GW events sorted by id."""
    return sorted(bootstrap["events"], key=lambda e: e["id"])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def backfill(target_gw: int | None = None) -> None:
    """Backfill historical snapshots using the Wayback Machine."""
    print("Fetching live bootstrap to get GW deadline schedule...")
    try:
        live = fetch_live_bootstrap()
    except Exception as e:
        print(f"ERROR: Could not fetch live bootstrap: {e}")
        return
    current_gw_id = next(
        (e["id"] for e in live["events"] if e.get("is_current")), None
    )
    if current_gw_id is None:
        print("ERROR: Could not determine current GW from live bootstrap.")
        return
    print(f"Current GW: {current_gw_id}")

    events = gw_events(live)
    past_events = [e for e in events if e["id"] < current_gw_id and e.get("finished", False)]

    if target_gw is not None:
        past_events = [e for e in past_events if e["id"] == target_gw]
        if not past_events:
            print(f"ERROR: GW{target_gw} not found or is not a finished past GW.")
            return

    print(f"GWs to backfill: {[e['id'] for e in past_events]}\n")

    ok, skipped, failed = 0, 0, 0

    for event in past_events:
        gw = event["id"]
        deadline = event["deadline_time"]
        out_path = SNAPSHOTS_DIR / f"bootstrap_gw{gw}.json"

        if out_path.exists():
            print(f"GW{gw:02d}  SKIP  (already cached at {out_path})")
            skipped += 1
            continue

        print(f"GW{gw:02d}  deadline={deadline}")
        try:
            ts = find_wayback_snapshot(deadline)
            if ts is None:
                print(f"  WARNING: No Wayback snapshot found in [{LOOKBACK_DAYS}d before deadline, deadline)")
                failed += 1
                continue

            time.sleep(WAYBACK_DELAY)
            bootstrap = fetch_wayback_bootstrap(ts)
            path = save_snapshot(bootstrap, gw)
            print(f"  SAVED  {path}")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

        time.sleep(WAYBACK_DELAY)

    print(f"\nDone. saved={ok}  skipped={skipped}  failed={failed}")


def live_mode() -> None:
    """Fetch the current live bootstrap and cache it for the current GW."""
    try:
        bootstrap = fetch_live_bootstrap()
    except Exception as e:
        print(f"ERROR: Could not fetch live bootstrap: {e}")
        return
    current_gw = next(
        (e["id"] for e in bootstrap["events"] if e.get("is_current")), None
    )
    next_gw = next(
        (e["id"] for e in bootstrap["events"] if e.get("is_next")), None
    )

    # Save snapshot for current GW (post-deadline, pre-GW results)
    # and also for next GW (pre-deadline planning snapshot).
    saved = []
    for gw in filter(None, set([current_gw, next_gw])):
        path = save_snapshot(bootstrap, gw)
        saved.append(str(path))

    print(f"Live bootstrap saved for GW(s): {saved}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gw", type=int, default=None, help="Backfill a single GW")
    parser.add_argument("--live", action="store_true", help="Fetch live bootstrap (for cron)")
    args = parser.parse_args()

    if args.live:
        live_mode()
    else:
        backfill(target_gw=args.gw)


if __name__ == "__main__":
    main()
