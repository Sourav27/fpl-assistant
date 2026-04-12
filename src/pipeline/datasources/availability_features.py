"""Unified availability feature assembly — single entry point for all availability logic.

Produces four feature columns per player:
    is_injured              — FPL status in {i, u}
    is_suspended            — FPL status == 's'
    is_doubt                — FPL status == 'd' (NOT driven by chance threshold)
    signal_confidence       — weighted average of agreeing sources
    n_corroborating_sources — count of non-FPL sources that agree with primary signal

Source weight table:
    FPL news/status:     1.0 (primary)
    premierinjuries.com: 0.8 (fallback when FPL status='a' and no FPL news)
    FFS RSS:             0.6 (corroboration only)
    Reddit r/FantasyPL:  0.5 (corroboration only)

The existing HybridAvailabilityFilter (availability.py) is not called here;
that module remains the xP-scaling layer. This module produces *features*
consumed downstream by the model.

Design notes:
    - is_doubt is status == 'd' only. chance_of_playing is a separate feature.
    - status='n' (not in squad) is excluded entirely (not flagged as injured).
    - status='s' (suspended) → is_suspended=1, is_injured=0.
    - Fallback to premierinjuries only when fpl_status='a' AND fpl_news is empty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Source weights for confidence aggregation
_SOURCE_WEIGHTS: dict[str, float] = {
    "fpl": 1.0,
    "premierinjuries": 0.8,
    "ffs": 0.6,
    "reddit": 0.5,
}

FplStatus = Literal["a", "d", "i", "u", "s", "n"]


@dataclass
class AvailabilitySignal:
    """Raw availability signal from a single source."""

    source: str  # one of: "fpl", "premierinjuries", "ffs", "reddit"
    is_injured: bool = False
    is_doubt: bool = False
    # is_suspended is FPL-only — not exposed per secondary source
    is_suspended: bool = False


@dataclass
class AvailabilityFeatures:
    """Computed availability features for a single player."""

    is_injured: int = 0
    is_suspended: int = 0
    is_doubt: int = 0
    signal_confidence: float = 0.0
    n_corroborating_sources: int = 0


def compute_availability_features(
    fpl_status: FplStatus,
    fpl_news: str,
    premierinjuries_status: str | None = None,
    secondary_signals: list[AvailabilitySignal] | None = None,
) -> AvailabilityFeatures:
    """Compute availability features from all available sources.

    Args:
        fpl_status:              FPL element status character ('a','d','i','u','s','n').
        fpl_news:                FPL news string (empty string = no news).
        premierinjuries_status:  'injured', 'doubt', or None (not listed = available).
        secondary_signals:       Optional list of AvailabilitySignal from FFS/Reddit.

    Returns:
        AvailabilityFeatures with five scalar fields.
    """
    if secondary_signals is None:
        secondary_signals = []

    # ── Step 1: derive primary signal from FPL ─────────────────────────────────
    is_injured_primary = int(fpl_status in {"i", "u"})
    is_suspended_primary = int(fpl_status == "s")
    # is_doubt: driven by status == 'd' only — NOT by chance threshold
    is_doubt_primary = int(fpl_status == "d")

    # ── Step 2: fallback — premierinjuries when FPL has no signal ─────────────
    # Apply only when: status='a' AND fpl_news is empty
    is_injured = is_injured_primary
    is_doubt = is_doubt_primary
    used_premierinjuries_fallback = False

    if fpl_status == "a" and not fpl_news.strip() and premierinjuries_status is not None:
        is_injured = int(premierinjuries_status == "injured")
        is_doubt = int(premierinjuries_status == "doubt")
        used_premierinjuries_fallback = True
        if is_injured or is_doubt:
            logger.debug(
                "availability_features: premierinjuries fallback triggered "
                "(status='a', no FPL news) → is_injured=%d is_doubt=%d",
                is_injured,
                is_doubt,
            )

    # ── Step 3: corroboration — count secondary sources that agree ────────────
    primary_flagged = bool(is_injured or is_doubt or is_suspended_primary)
    corroborating_sources = []

    for sig in secondary_signals:
        source_flagged = sig.is_injured or sig.is_doubt
        if primary_flagged and source_flagged:
            corroborating_sources.append(sig.source)
        elif not primary_flagged and not source_flagged:
            # Both agree player is available — still counts as corroboration
            pass  # not counted toward injury/doubt corroboration

    # ── Step 4: signal_confidence ─────────────────────────────────────────────
    # Weighted average of *all agreeing* sources (FPL always included if flagged)
    agreeing_weights: list[float] = []

    if primary_flagged:
        agreeing_weights.append(_SOURCE_WEIGHTS["fpl"])
        if used_premierinjuries_fallback and (is_injured or is_doubt):
            agreeing_weights.append(_SOURCE_WEIGHTS["premierinjuries"])
        elif premierinjuries_status in {"injured", "doubt"} and primary_flagged:
            agreeing_weights.append(_SOURCE_WEIGHTS["premierinjuries"])

        for source in corroborating_sources:
            w = _SOURCE_WEIGHTS.get(source, 0.0)
            if w:
                agreeing_weights.append(w)

        confidence = sum(agreeing_weights) / len(agreeing_weights) if agreeing_weights else 0.0
    else:
        # Player available — no injury signal to be confident about
        confidence = 0.0

    return AvailabilityFeatures(
        is_injured=is_injured,
        is_suspended=is_suspended_primary,
        is_doubt=is_doubt,
        signal_confidence=round(confidence, 4),
        n_corroborating_sources=len(corroborating_sources),
    )
