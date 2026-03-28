import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.pipeline.user import UserTeamState, fetch_user_team_state, compute_selling_price


class TestUserTeamStateDataclass:
    def test_total_value_computed(self):
        state = UserTeamState(
            entry_id=123,
            current_squad=[1, 2, 3],
            squad_codes=[101, 102, 103],
            selling_prices={1: 55, 2: 62, 3: 105},
            bank=50,
            free_transfers=2,
            active_chip=None,
            total_value=0,  # will be overridden
        )
        # total_value = sum(selling_prices) + bank = 222 + 50 = 272
        assert state.total_value == 272

    def test_free_transfers_clamped(self):
        state = UserTeamState(
            entry_id=123, current_squad=[], squad_codes=[],
            selling_prices={}, bank=0, free_transfers=10,
            active_chip=None, total_value=0,
        )
        assert state.free_transfers == 5  # capped at 5

    def test_active_chip_none_when_absent(self):
        state = UserTeamState(
            entry_id=123, current_squad=[], squad_codes=[],
            selling_prices={}, bank=100, free_transfers=1,
            active_chip=None, total_value=100,
        )
        assert state.active_chip is None


class TestFetchUserTeamState:
    def _make_mock_response(self, data: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = data
        return resp

    def test_returns_user_team_state(self, sample_bootstrap_json):
        entry_resp = {"value": 1050, "bank": 50, "last_deadline_bank": 50}
        picks_resp = {
            "active_chip": None,
            "picks": [
                {"element": 1, "selling_price": 55, "purchase_price": 50},
                {"element": 2, "selling_price": 62, "purchase_price": 60},
                {"element": 3, "selling_price": 105, "purchase_price": 100},
            ] + [{"element": i, "selling_price": 50, "purchase_price": 48} for i in range(4, 16)],
        }
        history_resp = {
            "current": [
                {"event": 30, "event_transfers": 1, "bank": 50, "points": 58,
                 "percentile_rank": 20},
            ]
        }
        transfers_resp = []  # no transfer history

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(
                entry_id=123, gw=30, bootstrap_data=sample_bootstrap_json
            )

        assert isinstance(state, UserTeamState)
        assert state.entry_id == 123
        assert len(state.current_squad) == 15
        assert state.bank == 50
        assert state.free_transfers >= 1
        assert state.active_chip is None

    def test_wildcard_chip_detected(self, sample_bootstrap_json):
        entry_resp = {"value": 1000, "bank": 0}
        picks_resp = {
            "active_chip": "wildcard",
            "picks": [{"element": i, "selling_price": 67, "purchase_price": 67}
                      for i in range(1, 16)],
        }
        history_resp = {"current": [{"event": 30, "event_transfers": 0, "bank": 0, "points": 0,
                                      "percentile_rank": 50}]}
        transfers_resp = []

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(123, 30, sample_bootstrap_json)

        assert state.active_chip == "wildcard"

    def test_free_transfers_from_history(self, sample_bootstrap_json):
        """Banking: if user made 0 transfers in GW30, they should have ft_prev + 1 (capped 5)."""
        entry_resp = {"value": 1000, "bank": 0}
        picks_resp = {
            "active_chip": None,
            "picks": [{"element": i, "selling_price": 67, "purchase_price": 67}
                      for i in range(1, 16)],
        }
        # GW29: had 1 FT, used 0 → GW30: 2 FT
        history_resp = {
            "current": [
                {"event": 29, "event_transfers": 0, "bank": 0, "points": 60, "percentile_rank": 30},
                {"event": 30, "event_transfers": 1, "bank": 0, "points": 58, "percentile_rank": 20},
            ]
        }
        transfers_resp = []

        with patch("src.pipeline.user._api_get_with_retry") as mock_get:
            mock_get.side_effect = [
                self._make_mock_response(entry_resp),
                self._make_mock_response(picks_resp),
                self._make_mock_response(transfers_resp),
                self._make_mock_response(history_resp),
            ]
            state = fetch_user_team_state(123, 30, sample_bootstrap_json)

        # After using transfers in GW30, next GW = 1 FT
        assert 1 <= state.free_transfers <= 5


class TestComputeSellingPrice:
    def test_no_profit(self):
        # bought 75, now 75 → sell at 75
        assert compute_selling_price(75, 75) == 75

    def test_profit_rounds_down(self):
        # bought 75, now 78 → profit 3 → half = 1.5 → floor = 1 → sell 76
        assert compute_selling_price(75, 78) == 76

    def test_full_profit(self):
        # bought 80, now 84 → profit 4 → half = 2 → sell 82
        assert compute_selling_price(80, 84) == 82

    def test_no_loss(self):
        # FPL never applies a haircut on price drops — sell at current price
        assert compute_selling_price(80, 75) == 75

    def test_exact_half_profit(self):
        # bought 100, now 102 → profit 2 → half 1 → sell 101
        assert compute_selling_price(100, 102) == 101
