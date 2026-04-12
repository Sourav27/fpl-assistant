# tests/datasources/test_signals.py
import pytest
from src.pipeline.datasources.signals import PlayerSignal, resolve_player_name

MOCK_BOOTSTRAP = {
    "elements": [
        {"id": 1, "code": 80201, "web_name": "Salah",
         "first_name": "Mohamed", "second_name": "Salah", "team": 14},
        {"id": 2, "code": 54694, "web_name": "Wilson",
         "first_name": "Callum", "second_name": "Wilson", "team": 7},
        {"id": 3, "code": 99999, "web_name": "Wilson",
         "first_name": "Ben", "second_name": "Wilson", "team": 3},
    ]
}


def test_player_signal_fields():
    sig = PlayerSignal(
        player_code=80201,
        source="ffs",
        signal_type="doubt",
        text="Salah is a doubt for GW32.",
        timestamp="2026-04-05T08:00:00Z",
        confidence=0.9,
    )
    assert sig.player_code == 80201
    assert sig.source == "ffs"
    assert sig.signal_type == "doubt"


def test_resolve_exact_web_name():
    code = resolve_player_name("Salah", MOCK_BOOTSTRAP)
    assert code == 80201


def test_resolve_full_name():
    code = resolve_player_name("Mohamed Salah", MOCK_BOOTSTRAP)
    assert code == 80201


def test_resolve_ambiguous_returns_none():
    # Two players with web_name "Wilson" — must not resolve silently
    code = resolve_player_name("Wilson", MOCK_BOOTSTRAP)
    assert code is None


def test_resolve_unknown_returns_none():
    code = resolve_player_name("Nonexistent Player", MOCK_BOOTSTRAP)
    assert code is None


def test_log_unresolved_writes_csv(tmp_path):
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    log_unresolved_name("Unknown X", source="ffs", raw_text="Unknown X doubt", csv_path=csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Unknown X"
    assert df.iloc[0]["source"] == "ffs"


# ── Task 4: FFS tests (append after existing tests) ──────────────────────────
from src.pipeline.datasources.ffs import parse_ffs_feed, _classify_signal_type

MOCK_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Fantasy Football Scout</title>
    <item>
      <title>Salah doubt for GW32 after training knock</title>
      <description>Mohamed Salah is a doubt for the upcoming gameweek.</description>
      <pubDate>Sat, 05 Apr 2026 08:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Havertz available and fit to play</title>
      <description>Kai Havertz has returned to training and is available.</description>
      <pubDate>Sat, 05 Apr 2026 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""


def test_classify_doubt():
    assert _classify_signal_type("Salah doubt for GW32 after knock") == "doubt"


def test_classify_available():
    assert _classify_signal_type("Player available and fit to play") == "available"


def test_classify_injured():
    assert _classify_signal_type("Player ruled out for three weeks") == "injured"


def test_classify_general_news():
    assert _classify_signal_type("Manager press conference notes") == "general_news"


def test_parse_ffs_feed_returns_signals():
    signals = parse_ffs_feed(
        rss_content=MOCK_RSS_FEED,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    assert isinstance(signals, list)
    assert len(signals) > 0
    for sig in signals:
        from src.pipeline.datasources.signals import PlayerSignal
        assert isinstance(sig, PlayerSignal)
        assert sig.source == "ffs"


def test_parse_ffs_doubt_signal():
    signals = parse_ffs_feed(rss_content=MOCK_RSS_FEED, bootstrap_data=MOCK_BOOTSTRAP)
    doubt_signals = [s for s in signals if s.signal_type == "doubt"]
    assert len(doubt_signals) >= 1
    assert doubt_signals[0].player_code == 80201  # Salah


# ── Task 5: Reddit tests (append after FFS tests) ────────────────────────────
from src.pipeline.datasources.reddit import parse_reddit_posts

MOCK_REDDIT_RESPONSE = {
    "data": {
        "children": [
            {"data": {
                "title": "Salah doubtful — training ground reports suggest knock",
                "selftext": "Multiple sources saying Salah picked up a knock in training.",
                "created_utc": 1743840000,
                "score": 245,
            }},
            {"data": {
                "title": "My GW32 template — what do you think?",
                "selftext": "Here is my squad...",
                "created_utc": 1743840000,
                "score": 12,
            }},
        ]
    }
}


def test_parse_reddit_returns_signals():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=50,
    )
    assert isinstance(signals, list)
    assert all(hasattr(s, "player_code") for s in signals)


def test_parse_reddit_filters_low_score():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=300,
    )
    assert signals == []


def test_reddit_signals_have_low_confidence():
    signals = parse_reddit_posts(
        posts_data=MOCK_REDDIT_RESPONSE,
        bootstrap_data=MOCK_BOOTSTRAP,
        min_score=50,
    )
    for sig in signals:
        assert sig.confidence <= 0.6
        assert sig.source == "reddit"


# ── Task 6: premierinjuries tests (append after Reddit tests) ─────────────────
from src.pipeline.datasources.premierinjuries import (
    parse_premierinjuries_html,
    cross_verify_against_fpl,
)

MOCK_PI_HTML = """<html><body>
<table id="player-injury-table">
  <thead><tr><th>Player</th><th>Club</th><th>Status</th><th>Notes</th></tr></thead>
  <tbody>
    <tr><td>Mohamed Salah</td><td>Liverpool</td><td>Doubt</td><td>Knock in training</td></tr>
    <tr><td>Unknown Player X</td><td>City</td><td>Available</td><td>Fit to play</td></tr>
  </tbody>
</table>
</body></html>"""

MOCK_FPL_STATUS = {80201: "a"}


def test_parse_pi_html_returns_signals():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    assert isinstance(signals, list)
    assert all(s.source == "premierinjuries" for s in signals)


def test_parse_pi_resolved_signal():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    resolved = [s for s in signals if s.player_code == 80201]
    assert len(resolved) == 1
    assert resolved[0].signal_type == "doubt"


def test_parse_pi_unresolved_skipped():
    signals = parse_premierinjuries_html(
        html_content=MOCK_PI_HTML,
        bootstrap_data=MOCK_BOOTSTRAP,
    )
    player_codes = [s.player_code for s in signals]
    assert 80201 in player_codes
    assert len(signals) == 1


def test_parse_pi_unresolved_writes_csv(tmp_path):
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    log_unresolved_name(
        name="Unknown Player X", source="premierinjuries",
        raw_text="Unknown Player X: available.", csv_path=csv_path,
    )
    import pandas as pd
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["name"] == "Unknown Player X"
    assert df.iloc[0]["source"] == "premierinjuries"


def test_parse_reddit_unresolved_writes_csv(tmp_path):
    from src.pipeline.datasources.signals import log_unresolved_name
    csv_path = tmp_path / "signal_unresolved.csv"
    log_unresolved_name(
        name="Zxqwertyplayer123", source="reddit",
        raw_text="Zxqwertyplayer123 doubt for GW32", csv_path=csv_path,
    )
    import pandas as pd
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df.iloc[0]["source"] == "reddit"


def test_cross_verify_contradiction_detected():
    signal = PlayerSignal(
        player_code=80201, source="premierinjuries",
        signal_type="injured", text="Salah out", timestamp="2026-04-05T08:00:00Z",
    )
    fpl_status = {80201: "a"}
    result = cross_verify_against_fpl([signal], fpl_status)
    assert result[0]["contradicted"] is True


def test_cross_verify_consistent():
    signal = PlayerSignal(
        player_code=80201, source="premierinjuries",
        signal_type="doubt", text="Salah doubt", timestamp="2026-04-05T08:00:00Z",
    )
    fpl_status = {80201: "d"}
    result = cross_verify_against_fpl([signal], fpl_status)
    assert result[0]["contradicted"] is False


def test_source_column_map_importable():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    assert isinstance(SOURCE_COLUMN_MAP, dict)
    required_keys = {"fpl_post_gw", "fpl_pre_gw", "understat", "espn",
                     "fpl_news", "premierinjuries", "ffs", "reddit"}
    assert required_keys.issubset(SOURCE_COLUMN_MAP.keys())


def test_source_column_map_fpl_post_gw_has_required_columns():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    cols = SOURCE_COLUMN_MAP["fpl_post_gw"]["columns"]
    for col in ["minutes", "goals_scored", "assists", "expected_goals"]:
        assert col in cols
    assert "xg_chain" not in cols, "xg_chain must not be in fpl_post_gw — Understat owns it"
    assert "xg_buildup" not in cols, "xg_buildup must not be in fpl_post_gw — Understat owns it"


def test_source_column_map_understat_only_unique_cols():
    from src.pipeline.datasources import SOURCE_COLUMN_MAP
    cols = SOURCE_COLUMN_MAP["understat"]["columns"]
    assert cols == ["xg_chain", "xg_buildup"]
