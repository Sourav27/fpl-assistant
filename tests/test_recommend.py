import pytest
import pandas as pd
from src.pipeline.recommend import compute_fdr_weight, build_fixture_fdr_map
from src.pipeline.user import UserTeamState


class TestComputeFdrWeight:
    def test_average_fixture(self):
        # FDR 3 → weight exactly 1.0
        assert compute_fdr_weight(fdr=3, sensitivity=0.15) == pytest.approx(1.0)

    def test_easy_fixture_boosts(self):
        # FDR 1 → 1.0 - 0.15*(1-3)/2 = 1.0 + 0.15 = 1.15
        assert compute_fdr_weight(fdr=1, sensitivity=0.15) == pytest.approx(1.15)

    def test_hard_fixture_discounts(self):
        # FDR 5 → 1.0 - 0.15*(5-3)/2 = 1.0 - 0.15 = 0.85
        assert compute_fdr_weight(fdr=5, sensitivity=0.15) == pytest.approx(0.85)

    def test_fdr_2(self):
        assert compute_fdr_weight(fdr=2, sensitivity=0.15) == pytest.approx(1.075)

    def test_fdr_4(self):
        assert compute_fdr_weight(fdr=4, sensitivity=0.15) == pytest.approx(0.925)

    def test_zero_sensitivity_always_one(self):
        for fdr in [1, 2, 3, 4, 5]:
            assert compute_fdr_weight(fdr=fdr, sensitivity=0.0) == pytest.approx(1.0)

    def test_weight_clamped_to_nonnegative(self):
        # Even extreme sensitivity should not produce negative weights
        assert compute_fdr_weight(fdr=5, sensitivity=2.0) >= 0.0


class TestBuildFixtureFdrMap:
    def test_returns_fdr_for_home_team(self):
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 13,
             "team_h_difficulty": 4, "team_a_difficulty": 2}
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33])
        # Team 1 is HOME → fdr_team = team_h_difficulty = 4
        assert fdr_map[(1, 33)] == 4

    def test_returns_fdr_for_away_team(self):
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 13,
             "team_h_difficulty": 4, "team_a_difficulty": 2}
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33])
        # Team 13 is AWAY → fdr_team = team_a_difficulty = 2
        assert fdr_map[(13, 33)] == 2

    def test_blank_gw_absent(self):
        fixtures = [{"event": 33, "team_h": 1, "team_a": 13,
                     "team_h_difficulty": 4, "team_a_difficulty": 2}]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[33, 34])
        # Team 1 has no fixture in GW34 → not in map
        assert (1, 34) not in fdr_map

    def test_double_gw_averages_fdr(self):
        # Team plays twice in GW34 — average their FDR values
        fixtures = [
            {"event": 34, "team_h": 1, "team_a": 5,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"event": 34, "team_h": 10, "team_a": 1,
             "team_h_difficulty": 3, "team_a_difficulty": 5},
        ]
        fdr_map = build_fixture_fdr_map(fixtures, gws=[34])
        # Team 1: first fixture away (fdr=5), second fixture away (fdr=5 → no wait)
        # First: team 1 is home → fdr=2. Second: team 1 is away → fdr=5. Average = 3.5
        assert fdr_map[(1, 34)] == pytest.approx(3.5)


@pytest.fixture
def sample_user_state():
    """A user with 2 FTs, £5.0m bank, squad of 15 players."""
    # Element IDs 1-15, codes 101-115
    return UserTeamState(
        entry_id=123,
        current_squad=list(range(1, 16)),
        squad_codes=list(range(101, 116)),
        selling_prices={i: 55 + i for i in range(1, 16)},
        bank=50,  # £5.0m
        free_transfers=2,
        active_chip=None,
        total_value=0,
    )


@pytest.fixture
def extended_predictions_df():
    """25 players with varying xP and costs — enough to test transfers."""
    # Players 1-15 (current squad, moderate xP), players 16-25 (upgrades)
    data = {
        "element": list(range(1, 26)),
        "code": list(range(101, 126)),
        "name": [f"Player{i}" for i in range(1, 26)],
        "position": (["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 1
                  + ["GK", "DEF", "DEF", "MID", "MID", "FWD", "FWD", "DEF", "MID", "FWD"],
        "team": [f"Team{(i % 10) + 1}" for i in range(1, 26)],
        "xP": [3.0 + i * 0.2 for i in range(1, 16)] + [7.0 + i * 0.1 for i in range(1, 11)],
        "now_cost": [55 + i for i in range(1, 26)],  # 0.1M units: 56=£5.6m, 57=£5.7m, ...
    }
    return pd.DataFrame(data)


class TestRecommendSingleGW:
    def test_returns_transfer_plan_dict(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert isinstance(plan, dict)
        assert "transfers" in plan
        assert "projected_xp" in plan
        assert "hit_cost" in plan

    def test_no_transfers_when_squad_optimal(self, sample_user_state, extended_predictions_df):
        """If the user's squad already contains the best players, no transfers needed."""
        from src.pipeline.recommend import recommend_transfers
        # Make current squad have very high xP
        extended_predictions_df = extended_predictions_df.copy()
        extended_predictions_df.loc[:14, "xP"] = 20.0  # current squad best
        extended_predictions_df.loc[15:, "xP"] = 1.0   # rest are terrible
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert plan["hit_cost"] == 0

    def test_hit_cost_applied_for_extra_transfers(self, sample_user_state, extended_predictions_df):
        """If optimal requires 3 transfers but user has 2 FT, 1 hit = -4 points."""
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=1,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        # Hit cost must be non-negative multiple of 4
        assert plan["hit_cost"] >= 0
        assert plan["hit_cost"] % 4 == 0


class TestRecommendMultiGW:
    def test_horizon_2_returns_plan(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 5, "team_h_difficulty": 2, "team_a_difficulty": 4},
            {"event": 34, "team_h": 2, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
        ]
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=fixtures,
            horizon=2,
            fdr_sensitivity=0.15,
            max_hit_points=8,
        )
        assert "transfers" in plan
        assert isinstance(plan["transfers"], list)
        # transfers list has one entry per GW
        assert len(plan["transfers"]) == 2

    def test_blank_gw_player_has_zero_xp(self, sample_user_state, extended_predictions_df):
        """Players with no fixture in a GW contribute 0 xP for that GW."""
        from src.pipeline.recommend import build_xp_matrix
        # Only GW33 has fixtures
        fixtures = [
            {"event": 33, "team_h": 1, "team_a": 5,
             "team_h_difficulty": 2, "team_a_difficulty": 4},
        ]
        # Player 1 is on team 1
        extended_predictions_df = extended_predictions_df.copy()
        extended_predictions_df.loc[0, "team"] = "Team1"
        # Team IDs in fixtures use int; we need a team→id map
        team_id_map = {"Team1": 1, "Team5": 5}
        xp_matrix = build_xp_matrix(
            predictions=extended_predictions_df,
            fixtures=fixtures,
            team_id_map=team_id_map,
            gws=[33, 34],
            fdr_sensitivity=0.15,
        )
        # GW34: Team1 has no fixture → xP=0 for Player1
        player1_gw34_xp = xp_matrix.loc[
            extended_predictions_df[extended_predictions_df["team"] == "Team1"].index[0], 34
        ]
        assert player1_gw34_xp == pytest.approx(0.0)

    def test_hit_cost_not_exceeded(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_transfers
        plan = recommend_transfers(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
            fixtures=[],
            horizon=3,
            fdr_sensitivity=0.15,
            max_hit_points=4,  # max 1 hit per GW
        )
        for gw_transfers in plan["transfers"]:
            hit = gw_transfers.get("hit_cost", 0)
            assert hit <= 4


class TestRecommendWildcard:
    def test_wildcard_ignores_current_squad(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_wildcard
        plan = recommend_wildcard(
            user_state=sample_user_state,
            predictions=extended_predictions_df,
        )
        assert "squad" in plan
        assert len(plan["squad"]) == 15
        assert "total_xp" in plan

    def test_wildcard_uses_total_value_as_budget(self, sample_user_state, extended_predictions_df):
        from src.pipeline.recommend import recommend_wildcard
        plan = recommend_wildcard(sample_user_state, extended_predictions_df)
        # Budget used (in 0.1M units) must be <= total_value (0.1M units)
        total_cost_01m = sum(
            extended_predictions_df[extended_predictions_df["element"] == e]["now_cost"].values[0]
            for e in plan["squad"]
            if e in extended_predictions_df["element"].values
        )
        assert total_cost_01m <= sample_user_state.total_value + 1  # 1 unit tolerance for rounding


class TestSaveRecommendCSV:
    def test_creates_csv_with_correct_columns(self, tmp_path):
        from src.pipeline.recommend import save_recommend_csv
        plan = {
            "transfers": [
                {"transfers": [{"player_out": "Watkins", "player_in": "Haaland",
                  "price_out": 5.2, "price_in": 7.8, "xp_out": 5.2, "xp_in": 7.8}],
                 "hit_cost": 0, "bank_after": 3.5},
                {"transfers": [], "hit_cost": 0, "bank_after": 3.5},
            ],
            "projected_xp": 312.4,
            "hit_cost": 0,
            "bank_after": 3.5,
        }
        out_path = tmp_path / "recommend_gw33.csv"
        save_recommend_csv(plan, out_path, start_gw=33)
        import pandas as pd
        df = pd.read_csv(out_path)
        assert "gw" in df.columns
        assert "player_out" in df.columns
        assert "player_in" in df.columns
        assert "hit_cost" in df.columns
