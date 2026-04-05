# src/pipeline/datasources/signals.py
"""Shared signal dataclass and player name resolver."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerSignal:
    player_code: int           # FPL persistent player code
    source: str                # "ffs" | "reddit" | "premierinjuries"
    signal_type: str           # "doubt" | "available" | "injured" | "rotation_risk" | "differential"
    text: str                  # Raw signal text
    timestamp: str             # ISO 8601
    confidence: float = 1.0   # 0–1; 1.0 = structured source, lower = NLP-inferred


def resolve_player_name(name: str, bootstrap_data: dict) -> int | None:
    """Resolve a player name string to an FPL persistent player code.

    Resolution order:
    1. Exact match on web_name
    2. Exact match on first_name + ' ' + second_name
    3. Returns None (ambiguous or unresolved) — never guesses

    Returns None if ambiguous (multiple matches) or not found.
    """
    elements = bootstrap_data.get("elements", [])
    name_lower = name.strip().lower()

    # Step 1: exact web_name match
    web_matches = [e for e in elements if e["web_name"].lower() == name_lower]
    if len(web_matches) == 1:
        return web_matches[0]["code"]
    if len(web_matches) > 1:
        logger.warning("Ambiguous web_name '%s' — %d matches, skipping", name, len(web_matches))
        return None

    # Step 2: full name match
    full_matches = [
        e for e in elements
        if (e["first_name"] + " " + e["second_name"]).lower() == name_lower
    ]
    if len(full_matches) == 1:
        return full_matches[0]["code"]
    if len(full_matches) > 1:
        logger.warning("Ambiguous full name '%s' — %d matches, skipping", name, len(full_matches))
        return None

    return None


def log_unresolved_name(
    name: str,
    source: str,
    raw_text: str,
    csv_path: Path | None = None,
    timestamp: str = "",
) -> None:
    """Write an unresolved player name to signal_unresolved.csv for manual review."""
    import pandas as pd
    if csv_path is None:
        from src.config import SIGNAL_UNRESOLVED_CSV
        csv_path = SIGNAL_UNRESOLVED_CSV
    csv_path = Path(csv_path)
    row = pd.DataFrame([{
        "name": name,
        "source": source,
        "raw_text": raw_text,
        "timestamp": timestamp,
    }])
    if csv_path.exists():
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(csv_path, index=False)
    logger.debug("Unresolved name '%s' from '%s' logged to %s", name, source, csv_path)
