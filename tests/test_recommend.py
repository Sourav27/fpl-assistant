import pytest
from src.pipeline.recommend import compute_fdr_weight, build_fixture_fdr_map


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
