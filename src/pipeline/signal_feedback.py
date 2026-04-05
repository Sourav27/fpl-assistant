"""Signal feedback logger — tracks prediction accuracy per source-type pair.

When team sheets arrive, log each PlayerSignal against actual lineup outcome.
Data feeds the Track G Phase 2 activation gate:
  Threshold: ≥ 80% accuracy over ≥ 15 observations per (source, signal_type) pair.

Schema: signal_id, source, signal_type, player_code, gw,
        predicted_status, actual_started, contradicted, run_date
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.pipeline.datasources.signals import PlayerSignal
from src.config import SIGNAL_ACCURACY_CSV

logger = logging.getLogger(__name__)


def make_signal_id(
    source: str, player_code: int, gw: int, signal_type: str, timestamp: str = ""
) -> str:
    """Deterministic signal ID: md5 of key fields + timestamp.

    Including timestamp prevents collision when the same source issues two signals
    for the same player/GW/type (e.g. two FFS articles both saying 'Salah doubt GW32').
    """
    raw = f"{source}:{player_code}:{gw}:{signal_type}:{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def append_signal_feedback(
    signal: PlayerSignal,
    gw: int,
    actual_started: bool,
    contradicted: bool,
    csv_path: Path | None = None,
) -> None:
    """Append one signal feedback row to signal_accuracy.csv.

    Args:
        signal: The PlayerSignal that was issued pre-deadline.
        gw: Gameweek number.
        actual_started: True if the player appeared in the starting XI.
        contradicted: True if signal contradicted FPL API status at issue time.
        csv_path: Override path (for testing). Defaults to SIGNAL_ACCURACY_CSV.
    """
    resolved_path: Path = Path(csv_path) if csv_path is not None else Path(SIGNAL_ACCURACY_CSV)

    row = pd.DataFrame([{
        "signal_id": make_signal_id(
            signal.source, signal.player_code, gw, signal.signal_type, signal.timestamp
        ),
        "source": signal.source,
        "signal_type": signal.signal_type,
        "player_code": signal.player_code,
        "gw": gw,
        "predicted_status": signal.signal_type,
        "actual_started": actual_started,
        "contradicted": contradicted,
        "run_date": datetime.now(tz=timezone.utc).isoformat(),
    }])

    if resolved_path.exists():
        row.to_csv(resolved_path, mode="a", header=False, index=False)
    else:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(resolved_path, index=False)


def compute_source_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(source, signal_type) accuracy from a feedback log DataFrame.

    Accuracy:
      - doubt/injured: correct if actual_started == False (player didn't start)
      - available: correct if actual_started == True

    Returns DataFrame with columns: source, signal_type, accuracy, n_observations
    """
    df = df.copy()
    df["correct"] = df.apply(
        lambda r: (not r["actual_started"])
        if r["signal_type"] in ("doubt", "injured")
        else bool(r["actual_started"]),
        axis=1,
    )
    return (
        df.groupby(["source", "signal_type"])
        .agg(accuracy=("correct", "mean"), n_observations=("correct", "count"))
        .reset_index()
    )
