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
