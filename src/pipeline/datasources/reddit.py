"""Reddit r/FantasyPL JSON API client → PlayerSignal list (display-only, Phase 1).

Endpoint: https://www.reddit.com/r/FantasyPL/new.json
No authentication required for public subreddits.
Rate limit: 1 req/sec — use User-Agent header.
Does NOT adjust xP — display-only until >= 80% accuracy over >= 15 observations.

Confidence = 0.5 (community source, lower than structured sources).
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
import requests
from .signals import PlayerSignal, resolve_player_name, log_unresolved_name
from .ffs import _classify_signal_type, _extract_player_names

logger = logging.getLogger(__name__)

REDDIT_URL = "https://www.reddit.com/r/FantasyPL/new.json"
REDDIT_USER_AGENT = "fpl-assistant-signals/1.0 (personal FPL tool)"


def fetch_reddit_posts(limit: int = 25, time_filter: str = "day") -> dict:
    """Fetch recent posts from r/FantasyPL (live HTTP)."""
    resp = requests.get(
        REDDIT_URL,
        params={"limit": limit, "t": time_filter},
        headers={"User-Agent": REDDIT_USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    time.sleep(1.0)  # Respect Reddit 1 req/sec rate limit
    return resp.json()  # type: ignore[no-any-return]


def parse_reddit_posts(
    posts_data: dict,
    bootstrap_data: dict,
    min_score: int = 50,
) -> list[PlayerSignal]:
    """Parse Reddit posts into PlayerSignal objects.

    Args:
        posts_data: JSON response from Reddit API (or mock for testing).
        bootstrap_data: FPL bootstrap for player name resolution.
        min_score: Minimum post score to consider.

    Returns:
        List of PlayerSignal objects. Confidence = 0.5 (community source).
    """
    signals = []
    for child in posts_data.get("data", {}).get("children", []):
        post = child.get("data", {})
        if post.get("score", 0) < min_score:
            continue

        title = post.get("title", "")
        body = post.get("selftext", "")
        combined = f"{title} {body}"
        signal_type = _classify_signal_type(combined)

        if signal_type == "general_news":
            continue

        ts = datetime.fromtimestamp(
            post.get("created_utc", 0), tz=timezone.utc
        ).isoformat()

        candidate_names = _extract_player_names(title)
        resolved_code = None
        for name in candidate_names:
            code = resolve_player_name(name, bootstrap_data)
            if code is not None:
                resolved_code = code
                break

        if resolved_code is None:
            log_unresolved_name(
                name=" / ".join(candidate_names[:2]) if candidate_names else title[:50],
                source="reddit",
                raw_text=title,
                timestamp=ts,
            )
            continue

        signals.append(PlayerSignal(
            player_code=resolved_code,
            source="reddit",
            signal_type=signal_type,
            text=title.strip(),
            timestamp=ts,
            confidence=0.5,
        ))

    return signals
