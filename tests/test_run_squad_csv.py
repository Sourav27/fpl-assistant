import pandas as pd
import pytest


def _make_optimize_result():
    xi = pd.DataFrame([
        {"element": 1, "name": "Salah",    "position": "MID", "team": "Liverpool", "xP": 10.0, "now_cost": 13.0},
        {"element": 2, "name": "Saka",     "position": "MID", "team": "Arsenal",   "xP": 8.0,  "now_cost": 10.0},
    ])
    bench = pd.DataFrame([
        {"element": 3, "name": "Flekken",  "position": "GK",  "team": "Brentford", "xP": 3.0,  "now_cost": 4.5},
        {"element": 4, "name": "Mykolenko","position": "DEF", "team": "Everton",   "xP": 2.0,  "now_cost": 4.0},
    ])
    return {
        "xi": xi, "bench": bench,
        "captain": xi.iloc[0], "vice_captain": xi.iloc[1],
        "squad": pd.concat([xi, bench]), "total_xp": 21.0,
    }


def test_build_squad_csv_columns():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    assert list(df.columns) == [
        "element", "name", "position", "team", "xP",
        "is_starter", "bench_order", "is_captain", "is_vice_captain", "now_cost"
    ]


def test_build_squad_csv_starters_have_null_bench_order():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    starters = df[df["is_starter"]]
    assert starters["bench_order"].isna().all()


def test_build_squad_csv_bench_ranked_by_xp_desc():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    bench = df[~df["is_starter"]].sort_values("bench_order")
    assert bench.iloc[0]["name"] == "Flekken"
    assert bench.iloc[0]["bench_order"] == 1


def test_build_squad_csv_captain_flags():
    from src.pipeline.run import _build_squad_csv
    df = _build_squad_csv(_make_optimize_result())
    assert df[df["element"] == 1].iloc[0]["is_captain"] is True
    assert df[df["element"] == 2].iloc[0]["is_vice_captain"] is True
    assert df[df["element"] == 3].iloc[0]["is_captain"] is False


from unittest.mock import patch, MagicMock


def test_fetch_actual_transfers_filters_by_gw():
    from src.pipeline.run import _fetch_actual_transfers
    api_response = [
        {"element_in": 10, "element_out": 20, "element_in_cost": 85, "element_out_cost": 85,
         "event": 32, "time": "2026-04-10T10:00:00Z"},
        {"element_in": 11, "element_out": 21, "element_in_cost": 60, "element_out_cost": 65,
         "event": 31, "time": "2026-03-20T10:00:00Z"},
    ]
    bootstrap = {"elements": [
        {"id": 10, "web_name": "Saka"},
        {"id": 20, "web_name": "Salah"},
    ]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    with patch("src.pipeline.run._api_get_with_retry", return_value=mock_resp):
        result = _fetch_actual_transfers(entry_id=123, gw=32, bootstrap=bootstrap)
    assert len(result) == 1
    assert result[0]["gw"] == 32
    assert result[0]["player_in"] == "Saka"
    assert result[0]["player_out"] == "Salah"
    assert result[0]["hit_taken"] is False
    assert result[0]["transfer_rank"] == 1


def test_fetch_actual_transfers_hit_taken_when_costs_differ():
    from src.pipeline.run import _fetch_actual_transfers
    api_response = [
        {"element_in": 10, "element_out": 20, "element_in_cost": 85, "element_out_cost": 90,
         "event": 32, "time": "2026-04-10T10:00:00Z"},
    ]
    bootstrap = {"elements": [{"id": 10, "web_name": "A"}, {"id": 20, "web_name": "B"}]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    with patch("src.pipeline.run._api_get_with_retry", return_value=mock_resp):
        result = _fetch_actual_transfers(entry_id=1, gw=32, bootstrap=bootstrap)
    assert result[0]["hit_taken"] is True
