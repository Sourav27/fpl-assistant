# src/pipeline/availability.py
"""Player availability filtering — hybrid hard-exclude + soft-scale approach."""
import logging
import numpy as np
import pandas as pd
from src.config import (
    AVAILABILITY_HARD_EXCLUDE_STATUS,
    AVAILABILITY_HARD_EXCLUDE_CHANCE,
    AVAILABILITY_SOFT_SCALE,
)

logger = logging.getLogger(__name__)


def filter_availability(
    predictions: pd.DataFrame,
    bootstrap_data: dict,
) -> pd.DataFrame:
    """Filter and adjust predictions based on player availability.

    Decision table (first match wins):
      1. status in {i, u, s, n}           → hard exclude
      2. chance in {0, 25}                 → hard exclude
      3. chance == 50                      → xP * 0.50
      4. status == 'd' and chance is None  → xP * 0.50
      5. chance == 75                      → xP * 0.75
      6. status == 'a', chance 100/None    → no adjustment
      7. status == 'd', chance == 100      → no adjustment
    """
    # Build lookup: element_id → availability info
    avail_map = {}
    for el in bootstrap_data.get("elements", []):
        avail_map[el["id"]] = {
            "status": el.get("status", "a"),
            "chance": el.get("chance_of_playing_next_round"),
            "news": el.get("news", ""),
        }

    result = predictions.copy()

    status_s = result["element"].map(lambda e: avail_map.get(e, {}).get("status", "a"))
    chance_s = result["element"].map(lambda e: avail_map.get(e, {}).get("chance"))

    # Rules 1 & 2: hard exclude
    exclude_mask = (
        status_s.isin(list(AVAILABILITY_HARD_EXCLUDE_STATUS))
        | chance_s.isin(list(AVAILABILITY_HARD_EXCLUDE_CHANCE))
    )

    # Rules 3, 4, 5: soft scale — build all conditions with np.select
    scale_conditions = [
        *(chance_s == k for k in AVAILABILITY_SOFT_SCALE),
        (status_s == "d") & chance_s.isna(),  # doubtful + null chance → 50/50
    ]
    scale_choices = [*AVAILABILITY_SOFT_SCALE.values(), 0.50]
    scale_array: np.ndarray = np.select(scale_conditions, scale_choices, default=1.0)
    scale_factors = pd.Series(scale_array, index=result.index, dtype=float)

    if logger.isEnabledFor(logging.INFO):
        for idx in result.index[exclude_mask]:
            e = result.at[idx, "element"]
            info = avail_map.get(e, {})
            logger.info(f"Excluded {result.at[idx, 'name']} (status={info.get('status')}, news={info.get('news', '')})")
        for idx in result.index[scale_factors < 1.0]:
            if not exclude_mask.get(idx, False):
                logger.info(f"Scaled {result.at[idx, 'name']} xP by {scale_factors[idx]}")

    result = result[~exclude_mask].copy()
    result["xP"] = result["xP"].mul(scale_factors)  # type: ignore[arg-type]

    return result.reset_index(drop=True)  # type: ignore[return-value]
