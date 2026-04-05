"""Fantasy Football Scout RSS parser → PlayerSignal list.

Feed: https://www.fantasyfootballscout.co.uk/feed (standard RSS 2.0, updates hourly)
Covers: injuries, DGW/BGW confirmations, international duty minutes, team news.
Signal classification: rule-based keyword matching (not NLP).
Does NOT cover: non-EPL players, FPL chip strategy (editorial only).
"""
from __future__ import annotations
import logging
import re
import feedparser
from .signals import PlayerSignal, resolve_player_name, log_unresolved_name

logger = logging.getLogger(__name__)

FFS_FEED_URL = "https://www.fantasyfootballscout.co.uk/feed"

_DOUBT_KEYWORDS    = re.compile(r"\bdoubt|knock|concern|uncertain\b", re.I)
_AVAILABLE_KEYWORDS = re.compile(r"\bavailable|fit|returns|back in\b", re.I)
_INJURED_KEYWORDS  = re.compile(r"\bruled out|miss(?:es|ing)?|injured|out for\b", re.I)


def _classify_signal_type(text: str) -> str:
    """Classify signal type from title/description text using keyword rules.
    Priority: injured > doubt > available > general_news.
    """
    if _INJURED_KEYWORDS.search(text):
        return "injured"
    if _DOUBT_KEYWORDS.search(text):
        return "doubt"
    if _AVAILABLE_KEYWORDS.search(text):
        return "available"
    return "general_news"


def _extract_player_names(text: str) -> list[str]:
    """Extract candidate player names from a title string.
    Tries longest-first: 'Mohamed Salah' before 'Salah'.
    """
    stop_words = {"GW", "FPL", "Premier", "League", "Fantasy", "Football", "Scout",
                  "GW32", "GW33", "GW34", "GW35", "GW36", "GW37", "GW38"}
    tokens = text.split()
    candidates = [
        t.rstrip("'s,.")
        for t in tokens
        if t and t[0].isupper() and t not in stop_words
    ]
    names = []
    i = 0
    while i < len(candidates):
        if i + 1 < len(candidates):
            names.append(candidates[i] + " " + candidates[i + 1])
        names.append(candidates[i])
        i += 1
    return names


def parse_ffs_feed(
    rss_content: str | None = None,
    bootstrap_data: dict | None = None,
    url: str = FFS_FEED_URL,
) -> list[PlayerSignal]:
    """Parse the FFS RSS feed into PlayerSignal objects.

    Args:
        rss_content: Raw RSS XML string (for testing — bypasses HTTP fetch).
        bootstrap_data: FPL bootstrap dict for player name resolution.
        url: RSS feed URL (only used if rss_content is None).

    Returns:
        List of PlayerSignal objects. Unresolved names logged to signal_unresolved.csv.
    """
    feed = feedparser.parse(rss_content if rss_content else url)

    if not bootstrap_data:
        logger.warning("FFS: bootstrap_data not provided, skipping name resolution")
        return []

    signals = []
    for entry in feed.entries:
        title = entry.get("title", "")
        description = entry.get("summary", "")
        combined = f"{title} {description}"
        signal_type = _classify_signal_type(combined)
        timestamp = entry.get("published", "")

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
                source="ffs",
                raw_text=title,
                timestamp=timestamp,
            )
            continue

        signals.append(PlayerSignal(
            player_code=resolved_code,
            source="ffs",
            signal_type=signal_type,
            text=combined.strip(),
            timestamp=timestamp,
            confidence=0.9,
        ))

    return signals
