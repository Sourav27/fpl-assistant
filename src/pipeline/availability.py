# src/pipeline/availability.py
"""Player availability filtering — hybrid hard-exclude + soft-scale approach."""
import logging
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
    exclude_mask = pd.Series(False, index=result.index)
    scale_factors = pd.Series(1.0, index=result.index)

    for idx, row in result.iterrows():
        info = avail_map.get(row["element"])
        if info is None:
            continue

        status = info["status"]
        chance = info["chance"]

        # Rule 1: Hard exclude by status
        if status in AVAILABILITY_HARD_EXCLUDE_STATUS:
            exclude_mask[idx] = True
            logger.debug(f"Excluded {row['name']} (status={status}, news={info['news']})")
            continue

        # Rule 2: Hard exclude by chance
        if chance is not None and chance in AVAILABILITY_HARD_EXCLUDE_CHANCE:
            exclude_mask[idx] = True
            logger.debug(f"Excluded {row['name']} (chance={chance}%, news={info['news']})")
            continue

        # Rules 3 & 5: Soft scale by chance (50 → 0.50, 75 → 0.75)
        if chance is not None and chance in AVAILABILITY_SOFT_SCALE:
            scale_factors[idx] = AVAILABILITY_SOFT_SCALE[chance]
            logger.debug(f"Scaled {row['name']} xP by {AVAILABILITY_SOFT_SCALE[chance]} (chance={chance}%)")
            continue

        # Rule 4: Doubtful with null chance → treat as 50/50
        if status == "d" and chance is None:
            scale_factors[idx] = 0.50
            logger.debug(f"Scaled {row['name']} xP by 0.50 (doubtful, chance=null)")
            continue

        # Rules 5-7: No adjustment needed (chance=100/None with status=a/d)

    # Apply exclusions and scaling
    result = result[~exclude_mask].copy()
    scale_factors = scale_factors[~exclude_mask]
    result["xP"] = result["xP"] * scale_factors.values

    return result.reset_index(drop=True)
