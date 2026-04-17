import json
import pytest
import requests
from unittest.mock import patch, MagicMock, call
from src.pipeline.fetch import (
    fetch_bootstrap,
    fetch_player_history,
    fetch_fixtures,
    get_current_gw,
    get_next_deadline,
    extract_xp_snapshot,
    normalize_player_gw_to_vaastav,
    fetch_live_gw_data,
)


class TestFetchBootstrap:
    def test_returns_dict_with_expected_keys(self, sample_bootstrap_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_bootstrap_json

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_bootstrap()

        assert "elements" in data
        assert "events" in data
        assert "teams" in data

    def test_raises_on_http_error_after_retries(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = requests.RequestException("503 Server Error")

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp), \
             patch("src.pipeline.fetch.time.sleep"):  # skip retry delays
            with pytest.raises(requests.RequestException, match="503"):
                fetch_bootstrap()


class TestGetCurrentGW:
    def test_identifies_current_gw(self, sample_bootstrap_json):
        result = get_current_gw(sample_bootstrap_json)
        assert result == 30

    def test_returns_none_when_no_current(self):
        data = {"events": [{"id": 1, "is_current": False, "is_next": False}]}
        result = get_current_gw(data)
        assert result is None


class TestGetNextDeadline:
    def test_returns_next_deadline(self, sample_bootstrap_json):
        gw, deadline = get_next_deadline(sample_bootstrap_json)
        assert gw == 31
        assert "2026-03-20" in deadline


class TestExtractXpSnapshot:
    def test_returns_dict_of_id_to_xp(self, sample_bootstrap_json):
        result = extract_xp_snapshot(sample_bootstrap_json)
        assert result[1] == 4.2
        assert result[3] == 6.8

    def test_handles_none_ep_this(self):
        data = {"elements": [{"id": 1, "ep_this": None}]}
        result = extract_xp_snapshot(data)
        assert result[1] == 0.0


class TestFetchPlayerHistory:
    def test_returns_history_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "history": [{"round": 1, "total_points": 5}],
            "history_past": [],
        }

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_player_history(1)
        assert len(data["history"]) == 1


class TestFetchFixtures:
    def test_returns_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"id": 1, "event": 30, "team_h": 1, "team_a": 10,
             "team_h_difficulty": 3, "team_a_difficulty": 4}
        ]

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            data = fetch_fixtures()
        assert len(data) == 1
        assert data[0]["team_h_difficulty"] == 3


class TestNormalizePlayerGwToVaastav:
    def test_maps_api_fields_to_vaastav_schema(self, sample_bootstrap_json, sample_player_history_json):
        gw_row = sample_player_history_json["history"][0]
        bootstrap = sample_bootstrap_json

        result = normalize_player_gw_to_vaastav(gw_row, bootstrap)

        assert result["name"] == "Saka"
        assert result["position"] == "MID"
        assert result["team"] == "Arsenal"
        assert result["total_points"] == 8
        assert result["GW"] == 1
        assert result["season"] == "2025-26"
        assert result["element"] == 3

    def test_fills_unavailable_columns_with_nan(self, sample_bootstrap_json, sample_player_history_json):
        import math
        gw_row = sample_player_history_json["history"][0]
        result = normalize_player_gw_to_vaastav(gw_row, sample_bootstrap_json)

        # These columns are unavailable from API — must exist and be NaN
        assert "clearances_blocks_interceptions" in result
        assert math.isnan(result["clearances_blocks_interceptions"])
        assert "defensive_contribution" in result
        assert math.isnan(result["defensive_contribution"])


class TestFetchLiveGwData:
    # Bulk live endpoint format: {"elements": [{"id": N, "stats": {...}, "explain": [...]}]}
    _live_response = {
        "elements": [
            {
                "id": 3,
                "stats": {
                    "total_points": 8, "minutes": 90, "goals_scored": 1, "assists": 1,
                    "clean_sheets": 0, "goals_conceded": 2, "bonus": 3, "bps": 35,
                    "influence": "40.0", "creativity": "30.5", "threat": "45.0",
                    "ict_index": "11.5", "starts": 1,
                    "expected_goals": "0.25", "expected_assists": "0.15",
                    "expected_goal_involvements": "0.40", "expected_goals_conceded": "0.80",
                },
                "explain": [{"fixture": 10}],
            }
        ]
    }

    def test_returns_dataframe_for_one_gw(self, sample_bootstrap_json):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._live_response

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            result = fetch_live_gw_data(
                target_gw=1,
                bootstrap_data=sample_bootstrap_json,
                player_ids=[3],
            )

        assert len(result) == 1
        assert result.iloc[0]["name"] == "Saka"
        assert result.iloc[0]["GW"] == 1
        assert result.iloc[0]["total_points"] == 8

    def test_filters_by_player_ids(self, sample_bootstrap_json):
        """player_ids filter excludes players not in the list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._live_response

        with patch("src.pipeline.fetch.requests.get", return_value=mock_resp):
            result = fetch_live_gw_data(
                target_gw=1,
                bootstrap_data=sample_bootstrap_json,
                player_ids=[999],  # Saka is id=3, not in list
            )

        assert len(result) == 0

    def test_returns_empty_on_api_error(self, sample_bootstrap_json):
        """API failure returns empty DataFrame rather than raising."""
        import requests as _req
        with patch("src.pipeline.fetch.requests.get", side_effect=_req.RequestException("timeout")):
            result = fetch_live_gw_data(target_gw=1, bootstrap_data=sample_bootstrap_json)

        assert result.empty
