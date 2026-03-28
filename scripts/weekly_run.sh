#!/bin/bash
# scripts/weekly_run.sh — Cron-friendly FPL pipeline wrapper
#
# Schedule in crontab (adjust times per GW calendar):
#   GW32: 0 17 10 4 * /path/to/scripts/weekly_run.sh pre-deadline
#   GW32: 0 22 10 4 * /path/to/scripts/weekly_run.sh predict
#
# Or run manually:
#   ./scripts/weekly_run.sh pre-deadline
#   ./scripts/weekly_run.sh predict
#   ./scripts/weekly_run.sh post-gw
#   ./scripts/weekly_run.sh retrain

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

PHASE="${1:-full}"
LOGDIR="$PROJECT_DIR/logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/${PHASE}_${TIMESTAMP}.log"

echo "[$(date)] Starting phase: $PHASE" | tee "$LOGFILE"
python -m src.pipeline.run "$PHASE" "${@:2}" 2>&1 | tee -a "$LOGFILE"
echo "[$(date)] Completed phase: $PHASE" | tee -a "$LOGFILE"
