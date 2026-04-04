"""Tests for scripts/format_discord_results.py.

Culprit if failing: format_wildcard_xi_block() or format_my_team_block() in format_discord_results.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import format_discord_results as fdr

XI_ROWS = [
    {"element": 1,  "name": "Flekken",     "position": "GK",  "team": "BRE", "now_cost": 45.0, "xP": 4.1},
    {"element": 2,  "name": "Walker",      "position": "DEF", "team": "BUR", "now_cost": 44.0, "xP": 7.6},
    {"element": 3,  "name": "Virgil",      "position": "DEF", "team": "LIV", "now_cost": 63.0, "xP": 6.5},
    {"element": 4,  "name": "Cucurella",   "position": "DEF", "team": "CHE", "now_cost": 60.0, "xP": 6.3},
    {"element": 5,  "name": "B.Fernandes", "position": "MID", "team": "MUN", "now_cost": 103.0,"xP": 8.5},
    {"element": 6,  "name": "Semenyo",     "position": "MID", "team": "MCI", "now_cost": 82.0, "xP": 12.0},
    {"element": 7,  "name": "Amad",        "position": "MID", "team": "MUN", "now_cost": 62.0, "xP": 9.4},
    {"element": 8,  "name": "Hinshelwood", "position": "MID", "team": "BHA", "now_cost": 51.0, "xP": 9.8},
    {"element": 9,  "name": "Gomez",       "position": "MID", "team": "BHA", "now_cost": 49.0, "xP": 7.1},
    {"element": 10, "name": "G.Jesus",     "position": "FWD", "team": "ARS", "now_cost": 64.0, "xP": 7.1},
    {"element": 11, "name": "Mykolenko",   "position": "FWD", "team": "EVE", "now_cost": 49.0, "xP": 6.6},
]

SQUAD_REC_ROWS = XI_ROWS + [
    {"element": 12, "name": "Bayindir",  "position": "GK",  "team": "MUN", "now_cost": 47.0, "xP": 4.2},
    {"element": 13, "name": "Hume",      "position": "DEF", "team": "SUN", "now_cost": 45.0, "xP": 6.2},
    {"element": 14, "name": "Andersen",  "position": "DEF", "team": "CPL", "now_cost": 45.0, "xP": 5.0},
    {"element": 15, "name": "Welbeck",   "position": "FWD", "team": "BHA", "now_cost": 61.0, "xP": 5.7},
]

XI_REC_ROWS = XI_ROWS  # same starters for simplicity

REC_ROWS = [
    {"gw": 32, "action": "transfer", "player_out": "Wilson",   "player_in": "Semenyo",
     "price_out": 6.0, "price_in": 8.2, "xp_out": 1.0, "xp_in": 12.0, "hit_cost": 0, "bank_after": 1.5},
    {"gw": 32, "action": "transfer", "player_out": "Cunha",    "player_in": "Amad",
     "price_out": 8.0, "price_in": 6.2, "xp_out": 2.9, "xp_in": 9.4,  "hit_cost": 0, "bank_after": 1.5},
    {"gw": 33, "action": "transfer", "player_out": "Rashford", "player_in": "Salah",
     "price_out": 6.5, "price_in": 13.0,"xp_out": 3.0, "xp_in": 14.0, "hit_cost": 0, "bank_after": 0.0},
]


def test_wildcard_xi_contains_captain():
    """Player with highest xP must be marked (C)."""
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    assert "(C)" in block
    assert "Semenyo" in block.split("(C)")[0].split("\n")[-1]


def test_wildcard_xi_grouped_by_position():
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    for pos in ("GK", "DEF", "MID", "FWD"):
        assert pos in block


def test_wildcard_xi_captain_tie_picks_one():
    rows = [
        {"element": 1, "name": "A", "position": "MID", "team": "X", "now_cost": 60.0, "xP": 10.0},
        {"element": 2, "name": "B", "position": "MID", "team": "Y", "now_cost": 60.0, "xP": 10.0},
        {"element": 3, "name": "C", "position": "FWD", "team": "Z", "now_cost": 60.0, "xP": 7.0},
    ]
    block = fdr.format_wildcard_xi_block(rows, gw=32)
    assert block.count("(C)") == 1


def test_wildcard_xi_under_2000_chars():
    block = fdr.format_wildcard_xi_block(XI_ROWS, gw=32)
    assert len(block) < 2000


def test_my_team_shows_15_players():
    """My Team block must list all 15 players (11 starters + 4 bench)."""
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    player_lines = [l for l in block.split("\n") if l.strip().startswith("•")]
    assert len(player_lines) == 15


def test_my_team_shows_bench_header():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Bench" in block or "bench" in block


def test_my_team_shows_bank():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "1.5" in block


def test_my_team_shows_transfers():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Wilson" in block


def test_my_team_excludes_future_gw_transfers():
    """GW33 transfer (Rashford/Salah) must not appear in GW32 block."""
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert "Rashford" not in block
    assert "Salah" not in block


def test_my_team_under_2000_chars():
    block = fdr.format_my_team_block(SQUAD_REC_ROWS, XI_REC_ROWS, REC_ROWS, bank=1.5, gw=32)
    assert len(block) < 2000
