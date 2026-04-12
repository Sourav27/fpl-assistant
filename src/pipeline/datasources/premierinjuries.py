"""premierinjuries.com scraper -> PlayerSignal list (@BenDinnery content).

Covers: EPL player injury/availability status from structured HTML table.
Does NOT cover: non-EPL players, future fixture predictions.

Signal confidence = 0.8 (structured website, rule-based parse, but DOM can drift).
Cross-verification: contradictions with FPL API status are flagged as contradicted=True
and must NOT feed into xP adjustment — display-only until resolved.

FPL status values: "a"=available, "d"=doubt, "i"=injured, "u"=unavailable,
                   "s"=suspended, "n"=not available
"""
from __future__ import annotations
import logging
import requests
from bs4 import BeautifulSoup
from .signals import PlayerSignal, resolve_player_name, log_unresolved_name

logger = logging.getLogger(__name__)

PREMIERINJURIES_URL = "https://www.premierinjuries.com/injury-table.php"

_STATUS_MAP = {
    "doubt": "doubt",
    "50/50": "doubt",
    "injured": "injured",
    "out": "injured",
    "available": "available",
    "fit": "available",
    "recovered": "available",
}

_FPL_UNAVAILABLE = {"i", "u", "s", "n"}


def fetch_premierinjuries_html() -> str:
    """Fetch the premierinjuries.com injury table HTML."""
    resp = requests.get(
        PREMIERINJURIES_URL,
        headers={"User-Agent": "fpl-assistant-signals/1.0"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.text


def parse_premierinjuries_html(
    html_content: str,
    bootstrap_data: dict,
) -> list[PlayerSignal]:
    """Parse premierinjuries HTML into PlayerSignal objects.

    Looks for <table id="player-injury-table"> or first <table>.
    Unresolved player names are logged to signal_unresolved.csv.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", id="player-injury-table") or soup.find("table")
    if not table:
        logger.warning("premierinjuries: no table found in HTML")
        return []

    signals = []
    for row in table.find_all("tr")[1:]:  # Skip header row
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 3:
            continue
        player_name = cells[0]
        status_raw = cells[2].lower()
        notes = cells[3] if len(cells) > 3 else ""
        signal_type = _STATUS_MAP.get(status_raw, "general_news")

        code = resolve_player_name(player_name, bootstrap_data)
        if code is None:
            log_unresolved_name(
                name=player_name,
                source="premierinjuries",
                raw_text=f"{player_name}: {status_raw}. {notes}".strip(),
            )
            continue

        signals.append(PlayerSignal(
            player_code=code,
            source="premierinjuries",
            signal_type=signal_type,
            text=f"{player_name}: {status_raw}. {notes}".strip(),
            timestamp="",
            confidence=0.8,
        ))

    return signals


def cross_verify_against_fpl(
    signals: list[PlayerSignal],
    fpl_status: dict[int, str],
) -> list[dict]:
    """Cross-verify signals against FPL API status field.

    Returns list of {signal, contradicted: bool, fpl_status: str}.
    Contradicted signals must NOT feed into xP adjustment.
    """
    results = []
    for sig in signals:
        fpl_st = fpl_status.get(sig.player_code, "a")
        contradicted = False

        if sig.signal_type == "injured" and fpl_st == "a":
            contradicted = True
            logger.warning(
                "Contradiction: player %d marked injured but FPL says 'a'",
                sig.player_code,
            )
        elif sig.signal_type == "doubt" and fpl_st == "a":
            contradicted = True
            logger.warning(
                "Contradiction: player %d marked doubt but FPL says 'a'",
                sig.player_code,
            )
        elif sig.signal_type == "available" and fpl_st in _FPL_UNAVAILABLE:
            contradicted = True
            logger.warning(
                "Contradiction: player %d marked available but FPL says '%s'",
                sig.player_code,
                fpl_st,
            )

        results.append({"signal": sig, "contradicted": contradicted, "fpl_status": fpl_st})

    return results
