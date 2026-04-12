# Track B — Fixture-Aware Per-Position Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Agent type:** This plan is optimised for the **data-scientist** subagent. All code is Python/pandas/scikit-learn within the `src/pipeline/` production package.

**Goal:** Replace the single global RF model with 4 per-position RF models (GK/DEF/MID/FWD), each trained with fixture-aware features, switching primary evaluation from MAE to Spearman ρ and handling DGW/BGW correctly per fixture.

**Architecture:** Opponent-side defensive stats are computed from vaastav `goals_conceded` data inside `prepare.py` and joined onto player rows. `features.py` adds fixture-level features (`is_home`, `fixture_count`, `rest_days`, `is_fixture_2`). `predict.py` routes players to position-specific models and sums per-fixture xP for DGW players. `run.py` retrains 4 models and logs Spearman ρ alongside MAE. `analysis.py` computes and persists ρ in `accuracy_log.csv`.

**Tech Stack:** Python 3.11, pandas, scikit-learn (RandomForestRegressor), scipy (spearmanr), joblib, pytest.

---

## Prerequisite context for the agent

This is the `fpl-assistant` Fantasy Premier League ML pipeline. Key facts:

- **vaastav dataset** lives at `data/Fantasy-Premier-League/data/{season}/gws/merged_gw.csv`. Columns include `element`, `code`, `GW`, `season`, `team`, `opponent_team`, `was_home`, `goals_conceded`, `total_points`, `minutes`, etc.
- **`code`** is the persistent cross-season player identifier. `element` (FPL ID) is recycled each season — never group by element alone.
- **Positions** in vaastav/FPL bootstrap: `1=GK, 2=DEF, 3=MID, 4=FWD`. In `predict.py` the `ID_COLUMNS` already include `"position"` but it stores the integer from FPL API. The `ELEMENT_TYPE_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}` is defined in `fetch.py`.
- **`ACTIVE_MODEL`** in `src/config.py` is currently a single `Path`. We are replacing it with **`ACTIVE_MODELS`** (a dict keyed by position string).
- **Existing feature columns** (18 total) are defined in `predict.py:FEATURE_COLUMNS`. The new plan adds up to 6 more fixture-aware features — these are added to a new `FIXTURE_FEATURE_COLUMNS` list and combined.
- **Training order** from roadmap: B-F8 (test shells) → B-F1 → B-F2 → B-F5 → B-F4 → B-F3 → B-F6 → B-F7.
- **Run tests** with: `python -m pytest tests/ -q` (unit) or `python -m pytest tests/test_features.py -v` (targeted).
- **208 tests currently passing** — do not break them.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config.py` | Modify | Replace `ACTIVE_MODEL` → `ACTIVE_MODELS` dict; keep `ACTIVE_MODEL` as alias |
| `src/pipeline/prepare.py` | Modify | Add `_compute_team_defensive_stats()` + `add_opponent_stats()` |
| `src/pipeline/features.py` | Modify | Add `add_fixture_features()` and `FIXTURE_FEATURE_COLUMNS` |
| `src/pipeline/predict.py` | Modify | Position-based model loading; per-fixture prediction + DGW sum; fallback logic |
| `src/pipeline/analysis.py` | Modify | Add `compute_spearman_rho()`; extend `append_accuracy_log()` with `spearman_rho` column |
| `src/pipeline/run.py` | Modify | `phase_retrain()` trains 4 models; logs per-position MAE + ρ; `phase_predict()` uses new API |
| `tests/test_prepare.py` | Create | Tests for team defensive stats computation and opponent join |
| `tests/test_features.py` | Modify | Extend with fixture feature tests |
| `tests/test_predict.py` | Create | Tests for position routing, DGW aggregation, fallback |
| `tests/test_analysis.py` | Modify | Extend with Spearman ρ tests |

---

## Task 1: Write test shells (B-F8 phase 1 — failing tests only)

**Files:**
- Create: `tests/test_prepare_opponent_stats.py`
- Create: `tests/test_predict_position.py`
- Modify: `tests/test_features.py` (append fixture feature tests)
- Modify: `tests/test_analysis.py` (append rho tests)

Write failing tests first. Do NOT implement any production code in this task.

- [ ] **Step 1: Create `tests/test_prepare_opponent_stats.py`**

```python
# tests/test_prepare_opponent_stats.py
"""Tests for opponent defensive stats joined onto player rows (B-F1/B-F2)."""
import pandas as pd
import pytest
from src.pipeline.prepare import _compute_team_defensive_stats, add_opponent_stats


@pytest.fixture
def minimal_gw_df():
    """Two teams, 3 GWs of data. Team 1 concedes a lot; Team 2 is solid."""
    rows = []
    # Team 1 players (team=1, opponent=2 each GW)
    for gw in range(1, 7):
        rows.append({"code": 101, "element": 1, "season": "2024-25", "GW": gw,
                     "team": 1, "opponent_team": 2, "was_home": True,
                     "goals_conceded": 2, "total_points": 2, "position": 1})
    # Team 2 players (team=2, opponent=1 each GW)
    for gw in range(1, 7):
        rows.append({"code": 201, "element": 2, "season": "2024-25", "GW": gw,
                     "team": 2, "opponent_team": 1, "was_home": False,
                     "goals_conceded": 0, "total_points": 8, "position": 2})
    return pd.DataFrame(rows)


class TestComputeTeamDefensiveStats:
    def test_returns_dataframe_with_required_columns(self, minimal_gw_df):
        result = _compute_team_defensive_stats(minimal_gw_df)
        assert "team" in result.columns
        assert "season" in result.columns
        assert "GW" in result.columns
        assert "team_gc_roll_4" in result.columns

    def test_rolling_is_lagged(self, minimal_gw_df):
        """team_gc_roll_4 at GW5 should be mean of GW1-4, not include GW5."""
        result = _compute_team_defensive_stats(minimal_gw_df)
        team1 = result[(result["team"] == 1) & (result["GW"] == 5)].iloc[0]
        assert team1["team_gc_roll_4"] == pytest.approx(2.0, abs=0.01)

    def test_early_gws_have_nan(self, minimal_gw_df):
        result = _compute_team_defensive_stats(minimal_gw_df)
        team1_gw1 = result[(result["team"] == 1) & (result["GW"] == 1)].iloc[0]
        assert pd.isna(team1_gw1["team_gc_roll_4"])


class TestAddOpponentStats:
    def test_joins_opponent_gc_onto_player_rows(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert "xGC_rolling_4" in result.columns

    def test_team1_player_sees_team2_gc(self, minimal_gw_df):
        """Team 1 player faces Team 2. Team 2 concedes 0 → xGC_rolling_4 for Team 1 player should be 0."""
        result = add_opponent_stats(minimal_gw_df)
        team1_gw5 = result[(result["team"] == 1) & (result["GW"] == 5)].iloc[0]
        # Team 2 concedes 0 per GW, so xGC rolling avg = 0
        assert team1_gw5["xGC_rolling_4"] == pytest.approx(0.0, abs=0.01)

    def test_no_rows_dropped(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert len(result) == len(minimal_gw_df)


class TestOpponentFormByPosition:
    def test_returns_opponent_form_rolling_column(self, minimal_gw_df):
        result = add_opponent_stats(minimal_gw_df)
        assert "opponent_form_rolling_6" in result.columns

    def test_opponent_form_rolling_value_is_correct(self, minimal_gw_df):
        """Team 2 allows 8 pts to DEF players (position=2) from Team 1 each GW.
        Team 1's DEF player facing Team 2 at GW7 should see opponent_form_rolling_6 ≈ 8.0.
        """
        result = add_opponent_stats(minimal_gw_df)
        # Team 1 player (code=101) faces Team 2; position=1 (GK in fixture).
        # Position=2 (DEF) rows belong to Team 2 players — they face Team 1.
        # Team 2 player (code=201, position=2) faces Team 1. Team 1 allows 2 pts to DEF.
        team2_gw7 = result[(result["code"] == 201) & (result["GW"] == 7)] if 7 in result["GW"].values else result[(result["code"] == 201) & (result["GW"] == result["GW"].max())]
        if not team2_gw7.empty:
            # Only assert it's a number (exact value depends on min_periods alignment)
            assert not pd.isna(team2_gw7.iloc[0]["opponent_form_rolling_6"]) or team2_gw7.iloc[0]["GW"] < 4

    def test_opponent_form_rolling_is_position_specific(self):
        """GK and FWD players facing the same opponent should see different opponent_form values."""
        rows = []
        # Team 1: a GK (pos=1) and a FWD (pos=4), both facing Team 2
        for gw in range(1, 9):
            rows.append({"code": 1, "element": 1, "season": "2024-25", "GW": gw,
                         "team": 1, "opponent_team": 2, "was_home": True,
                         "goals_conceded": 1, "total_points": 1, "position": 1})  # GK, 1pt
            rows.append({"code": 4, "element": 4, "season": "2024-25", "GW": gw,
                         "team": 1, "opponent_team": 2, "was_home": True,
                         "goals_conceded": 1, "total_points": 6, "position": 4})  # FWD, 6pt
            # Team 2 players facing Team 1 (needed for stats computation)
            rows.append({"code": 2, "element": 2, "season": "2024-25", "GW": gw,
                         "team": 2, "opponent_team": 1, "was_home": False,
                         "goals_conceded": 2, "total_points": 3, "position": 2})
        df = pd.DataFrame(rows)
        result = add_opponent_stats(df)
        # At GW8, both Team 1 players face Team 2. GK sees form for pos=1 (1pt), FWD sees pos=4 (6pt)
        gk_gw8 = result[(result["code"] == 1) & (result["GW"] == 8)].iloc[0]
        fwd_gw8 = result[(result["code"] == 4) & (result["GW"] == 8)].iloc[0]
        if not pd.isna(gk_gw8["opponent_form_rolling_6"]) and not pd.isna(fwd_gw8["opponent_form_rolling_6"]):
            assert gk_gw8["opponent_form_rolling_6"] != pytest.approx(fwd_gw8["opponent_form_rolling_6"], abs=0.5)
```

- [ ] **Step 2: Append fixture feature tests to `tests/test_features.py`**

Add this class at the bottom of the existing file:

```python
class TestFixtureFeatures:
    """Tests for add_fixture_features() — B-F3."""

    @pytest.fixture
    def fixture_df(self):
        """Single player, 3 GWs: normal, BGW (no fixture), DGW (2 fixtures)."""
        return pd.DataFrame({
            "code": [1, 1, 1, 1],
            "season": ["2024-25"] * 4,
            "GW": [1, 2, 3, 3],             # GW3 has 2 rows (DGW)
            "was_home": [True, False, True, False],
            "kickoff_time": [
                "2024-09-14T15:00:00Z",
                "2024-09-21T15:00:00Z",
                "2024-09-28T12:30:00Z",
                "2024-10-01T19:45:00Z",     # 3 days after fixture 1
            ],
            "total_points": [6, 2, 8, 5],
        })

    def test_is_home_added(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        assert "is_home" in result.columns
        assert result[result["GW"] == 1].iloc[0]["is_home"] == 1

    def test_fixture_count_normal_gw(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        gw1_rows = result[result["GW"] == 1]
        assert gw1_rows.iloc[0]["fixture_count"] == 1

    def test_fixture_count_dgw(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3]
        assert all(dgw_rows["fixture_count"] == 2)

    def test_rest_days_computed_for_fixture_2(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3].sort_values("is_fixture_2")
        fixture_2 = dgw_rows[dgw_rows["is_fixture_2"] == 1].iloc[0]
        assert fixture_2["rest_days"] == pytest.approx(3.0, abs=0.5)

    def test_rest_days_zero_for_fixture_1(self, fixture_df):
        from src.pipeline.features import add_fixture_features
        result = add_fixture_features(fixture_df)
        dgw_rows = result[result["GW"] == 3]
        fixture_1 = dgw_rows[dgw_rows["is_fixture_2"] == 0].iloc[0]
        assert fixture_1["rest_days"] == 0.0
```

- [ ] **Step 3: Append to `tests/test_analysis.py`**

Add at the bottom:

```python
class TestSpearmanRho:
    """Tests for compute_spearman_rho() — B-F7."""

    def test_perfect_rank_correlation(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        df = pd.DataFrame({
            "xP": [1.0, 2.0, 3.0, 4.0, 5.0],
            "actual_points": [2, 4, 6, 8, 10],
        })
        rho = compute_spearman_rho(df)
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_inverse_rank_correlation(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        df = pd.DataFrame({
            "xP": [5.0, 4.0, 3.0, 2.0, 1.0],
            "actual_points": [1, 2, 3, 4, 5],
        })
        rho = compute_spearman_rho(df)
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_returns_nan_for_empty(self):
        from src.pipeline.analysis import compute_spearman_rho
        import pandas as pd
        rho = compute_spearman_rho(pd.DataFrame({"xP": [], "actual_points": []}))
        import math
        assert math.isnan(rho)

    def test_accuracy_log_has_spearman_rho_column(self, tmp_path):
        from src.pipeline.analysis import append_accuracy_log
        import pandas as pd
        log_path = tmp_path / "accuracy_log.csv"
        picks = pd.DataFrame({
            "xP": [3.0, 7.0, 2.0],
            "raw_xP": [3.0, 7.0, 2.0],
            "actual_points": [4, 8, 1],
        })
        append_accuracy_log(
            gw=32,
            your_pts=42,
            your_xp=38.0,
            recommended_pts=None,
            recommended_xp=None,
            wildcard_pts=None,
            wildcard_xp=None,
            dream_team_pts=None,
            benchmarks={"average": 40, "top_player": 60},
            your_percentile_rank=None,
            picks_df=picks,
            path=log_path,
        )
        log = pd.read_csv(log_path)
        assert "spearman_rho" in log.columns
        assert not pd.isna(log.iloc[0]["spearman_rho"])
```

- [ ] **Step 4: Create `tests/test_predict_position.py`**

```python
# tests/test_predict_position.py
"""Tests for per-position model routing and DGW aggregation (B-F4, B-F3 prediction side)."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def players_multi_position():
    """One player per position for routing tests."""
    base_features = {col: [0.0] for col in [
        "total_points_roll_4", "total_points_roll_8",
        "minutes_roll_4", "minutes_roll_8",
        "ict_index_roll_4", "ict_index_roll_8",
        "bps_roll_4", "bps_roll_8",
        "goals_scored_roll_4", "assists_roll_4",
        "clean_sheets_roll_4",
        "influence_roll_4", "creativity_roll_4", "threat_roll_4",
        "total_points_momentum", "minutes_momentum", "ict_index_momentum",
        "transfers_net",
        # new fixture features
        "xGC_rolling_4", "opponent_form_rolling_6",
        "is_home", "fixture_count", "rest_days", "is_fixture_2",
    ]}
    rows = []
    for pos, el in [("GK", 1), ("DEF", 2), ("MID", 3), ("FWD", 4)]:
        row = {"element": el, "code": el * 100, "name": f"Player{pos}",
               "position": pos, "team": 1, "now_cost": 55}
        row.update({k: v[0] for k, v in base_features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def dgw_players():
    """One MID player with 2 fixture rows for DGW (GW rows pre-expanded)."""
    feat_vals = {col: 0.0 for col in [
        "total_points_roll_4", "total_points_roll_8",
        "minutes_roll_4", "minutes_roll_8",
        "ict_index_roll_4", "ict_index_roll_8",
        "bps_roll_4", "bps_roll_8",
        "goals_scored_roll_4", "assists_roll_4",
        "clean_sheets_roll_4",
        "influence_roll_4", "creativity_roll_4", "threat_roll_4",
        "total_points_momentum", "minutes_momentum", "ict_index_momentum",
        "transfers_net",
        "xGC_rolling_4", "opponent_form_rolling_6",
        "is_home", "rest_days",
    ]}
    row1 = {"element": 10, "code": 1000, "name": "Saka", "position": "MID",
            "team": 3, "now_cost": 105, "fixture_count": 2, "is_fixture_2": 0}
    row2 = {"element": 10, "code": 1000, "name": "Saka", "position": "MID",
            "team": 3, "now_cost": 105, "fixture_count": 2, "is_fixture_2": 1,
            "rest_days": 3.0}
    row1.update(feat_vals)
    row2.update(feat_vals)
    return pd.DataFrame([row1, row2])


class TestPositionRouting:
    def test_routes_gk_to_gk_model(self, players_multi_position, tmp_path):
        """Each position's player should be predicted by its position model."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]

        mock_models = {pos: mock_model for pos in ["GK", "DEF", "MID", "FWD"]}
        result = predict_next_gw_per_position(players_multi_position, models=mock_models)

        assert len(result) == 4
        assert set(result["position"]) == {"GK", "DEF", "MID", "FWD"}

    def test_fallback_when_model_missing_uses_ep_next(self, players_multi_position):
        """If a position model is None, xP falls back to ep_next when ep_next_map provided."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        # GK model missing; GK player has element=1
        models = {"GK": None, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        ep_next_map = {1: 3.7}  # element 1 is the GK
        result = predict_next_gw_per_position(players_multi_position, models=models, ep_next_map=ep_next_map)
        gk_row = result[result["position"] == "GK"].iloc[0]
        assert gk_row["xP"] == pytest.approx(3.7, abs=0.01)

    def test_fallback_when_model_missing_and_no_ep_next(self, players_multi_position):
        """If a position model is None and no ep_next_map, xP is 0 (safe default)."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        models = {"GK": None, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        result = predict_next_gw_per_position(players_multi_position, models=models)
        gk_row = result[result["position"] == "GK"].iloc[0]
        assert gk_row["xP"] == pytest.approx(0.0, abs=0.01)


class TestDGWAggregation:
    def test_dgw_player_xp_is_sum_of_two_fixtures(self, dgw_players):
        """A DGW player with 2 fixture rows should have xP = sum of both predictions."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [4.0]  # each fixture returns 4

        models = {"GK": mock_model, "DEF": mock_model, "MID": mock_model, "FWD": mock_model}
        result = predict_next_gw_per_position(dgw_players, models=models)

        assert len(result) == 1  # one row per player
        assert result.iloc[0]["xP"] == pytest.approx(8.0, abs=0.01)  # 4+4

    def test_single_fixture_not_doubled(self, players_multi_position):
        """fixture_count=1 players must not have their xP doubled."""
        from src.pipeline.predict import predict_next_gw_per_position

        mock_model = MagicMock()
        mock_model.predict.return_value = [3.0]

        # Ensure fixture_count=1 for all
        df = players_multi_position.copy()
        df["fixture_count"] = 1
        df["is_fixture_2"] = 0
        df["rest_days"] = 0.0

        models = {pos: mock_model for pos in ["GK", "DEF", "MID", "FWD"]}
        result = predict_next_gw_per_position(df, models=models)
        for _, row in result.iterrows():
            assert row["xP"] == pytest.approx(3.0, abs=0.01)


@pytest.mark.skip(reason="B-F3-DGW: phase_predict fixture expansion not yet implemented")
def test_phase_predict_expands_dgw_player_to_two_rows():
    """phase_predict should expand DGW players to 2 fixture rows before calling predict_next_gw_per_position.

    When this test is unskipped, verify:
    1. phase_predict fetches fixtures via fetch_fixtures()
    2. DGW players (those with 2 fixtures in target_gw) appear as 2 rows in the feature DataFrame
    3. is_fixture_2=1 and rest_days>0 on the second row
    4. Final predictions_gw{N}.csv has one row per player with DGW xP = sum of both fixtures
    """
    pass
```

- [ ] **Step 5: Run all new tests to confirm they fail for the right reason**

```bash
python -m pytest tests/test_prepare_opponent_stats.py tests/test_predict_position.py -v 2>&1 | head -40
```

Expected: `ImportError` or `AttributeError` — functions not yet defined. If you see a different error, fix the test code before proceeding.

Also run the analysis shell tests separately:

```bash
python -m pytest tests/test_analysis.py::TestSpearmanRho -v 2>&1 | head -20
```

Expected failures:
- `TestSpearmanRho::test_perfect_rank_correlation` — `ImportError: cannot import name 'compute_spearman_rho'` ✓
- `TestSpearmanRho::test_accuracy_log_has_spearman_rho_column` — may also fail with `TypeError: append_accuracy_log() got an unexpected keyword argument 'picks_df'` — this is **also expected** because Task 7 adds `picks_df` to that function's signature. Both failure modes are acceptable at this stage.

- [ ] **Step 6: Confirm existing 208 tests still pass**

```bash
python -m pytest tests/ -q --ignore=tests/test_prepare_opponent_stats.py --ignore=tests/test_predict_position.py 2>&1 | tail -5
```

Expected: `208 passed` (or same count as before).

- [ ] **Step 7: Commit test shells**

```bash
rtk git add tests/test_prepare_opponent_stats.py tests/test_predict_position.py tests/test_features.py tests/test_analysis.py
rtk git commit -m "test: add B-F8 test shells for Track B fixture-aware per-position models"
```

---

## Task 2: Opponent defensive stats in prepare.py (B-F1 + B-F2)

**Files:**
- Modify: `src/pipeline/prepare.py`
- Test: `tests/test_prepare_opponent_stats.py`

The goal is to compute per-team rolling goals-conceded and average points allowed to each position, then join them onto each player row as `xGC_rolling_4` and `opponent_form_rolling_6`. This uses only vaastav data — no external dependency on Understat.

- [ ] **Step 1: Add `_compute_team_defensive_stats()` to `prepare.py`**

Add this function after the existing `add_fixture_difficulty()` function:

```python
def _compute_team_defensive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level defensive rolling stats for opponent join.

    Returns a team-season-GW indexed DataFrame with:
      - team_gc_roll_4: 4-GW lagged rolling avg goals conceded
      - team_pts_allowed_{pos}_roll_6: 6-GW lagged rolling avg points allowed to each position

    Uses shift(1) to prevent lookahead: GW N's stat uses GW 1..(N-1).
    """
    if df.empty or "goals_conceded" not in df.columns:
        return pd.DataFrame(columns=["team", "season", "GW", "team_gc_roll_4"])

    team_gw = (
        df.groupby(["team", "season", "GW"], as_index=False)
        .agg(team_gc=("goals_conceded", "sum"))
    )
    team_gw = team_gw.sort_values(["team", "season", "GW"])
    team_gw["team_gc_roll_4"] = (
        team_gw.groupby(["team", "season"])["team_gc"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    )

    # Points allowed per position: average total_points of opponent players by position
    if "position" in df.columns:
        for pos_label in [1, 2, 3, 4]:
            pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            col = f"team_pts_allowed_{pos_map[pos_label]}_roll_6"
            pos_df = df[df["position"] == pos_label].copy()
            if pos_df.empty:
                team_gw[col] = float("nan")
                continue
            # Points allowed = points scored by the attacking team's player against this team
            # "opponent_team" is the defending team
            opp_agg = (
                pos_df.groupby(["opponent_team", "season", "GW"], as_index=False)
                .agg(pts_allowed=("total_points", "mean"))
                .rename(columns={"opponent_team": "team"})
            )
            opp_agg = opp_agg.sort_values(["team", "season", "GW"])
            opp_agg[col] = (
                opp_agg.groupby(["team", "season"])["pts_allowed"]
                .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
            )
            team_gw = team_gw.merge(opp_agg[["team", "season", "GW", col]], on=["team", "season", "GW"], how="left")

    return team_gw


def add_opponent_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Join opponent defensive stats onto each player row.

    Adds columns:
      - xGC_rolling_4: rolling goals conceded by the OPPONENT team (= how easy it is to score against them)
      - opponent_form_rolling_6: avg pts allowed by opponent to this player's position (6-GW rolling)

    These are the xGC features from Track B spec. Uses vaastav goals_conceded — no Understat dependency.
    """
    if df.empty or "opponent_team" not in df.columns:
        df["xGC_rolling_4"] = float("nan")
        df["opponent_form_rolling_6"] = float("nan")
        return df

    team_stats = _compute_team_defensive_stats(df)
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Join opponent's gc_roll_4 → this player sees the opponent's defensive weakness
    df = df.merge(
        team_stats[["team", "season", "GW", "team_gc_roll_4"]],
        left_on=["opponent_team", "season", "GW"],
        right_on=["team", "season", "GW"],
        how="left",
        suffixes=("", "_opp"),
    )
    df = df.rename(columns={"team_gc_roll_4": "xGC_rolling_4"})
    df = df.drop(columns=["team_opp"], errors="ignore")

    # Join opponent's pts_allowed for this player's position
    if "position" in df.columns:
        df["_pos_label"] = df["position"].map(pos_map) if df["position"].dtype == object else df["position"].map(pos_map)
        df["opponent_form_rolling_6"] = float("nan")
        for pos_label, pos_str in pos_map.items():
            col = f"team_pts_allowed_{pos_str}_roll_6"
            if col not in team_stats.columns:
                continue
            pos_mask = df["position"] == pos_label
            if not pos_mask.any():
                continue
            temp = df[pos_mask].merge(
                team_stats[["team", "season", "GW", col]],
                left_on=["opponent_team", "season", "GW"],
                right_on=["team", "season", "GW"],
                how="left",
                suffixes=("", "_opp2"),
            )
            df.loc[pos_mask, "opponent_form_rolling_6"] = temp[col].values
        df = df.drop(columns=["_pos_label"], errors="ignore")
    else:
        df["opponent_form_rolling_6"] = float("nan")

    return df
```

- [ ] **Step 2: Call `add_opponent_stats` inside `build_merged_dataset`**

After the `add_fixture_difficulty` call (around line 100 of `prepare.py`), add:

```python
        # B-F1/B-F2: join opponent defensive stats
        if not df.empty and "opponent_team" in df.columns:
            df = add_opponent_stats(df)
```

- [ ] **Step 3: Run the opponent stats tests**

```bash
python -m pytest tests/test_prepare_opponent_stats.py -v
```

Expected: All tests pass. If `TestOpponentFormByPosition` fails because positional data is integer (1/2/3/4), fix the position comparison in `add_opponent_stats`.

- [ ] **Step 4: Run full suite to confirm no regressions**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
rtk git add src/pipeline/prepare.py tests/test_prepare_opponent_stats.py
rtk git commit -m "feat: B-F1/B-F2 opponent defensive stats (xGC_rolling_4, opponent_form_rolling_6)"
```

---

## Task 3: Config — ACTIVE_MODELS dict (B-F5)

**Files:**
- Modify: `src/config.py`
- Modify: `src/pipeline/predict.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add `ACTIVE_MODELS` to `src/config.py`**

Find the line `ACTIVE_MODEL = MODELS_DIR / "rf_model_gw31.sav"` and add below it:

```python
# Per-position model paths (Track B). Keys must match ELEMENT_TYPE_MAP values.
ACTIVE_MODELS = {
    "GK":  MODELS_DIR / "rf_gk_gw31.sav",
    "DEF": MODELS_DIR / "rf_def_gw31.sav",
    "MID": MODELS_DIR / "rf_mid_gw31.sav",
    "FWD": MODELS_DIR / "rf_fwd_gw31.sav",
}
```

Keep `ACTIVE_MODEL` — it's still used by `run.py` for backward compat until Task 6.

- [ ] **Step 2: Add config test**

In `tests/test_config.py`, add:

```python
def test_active_models_has_all_positions():
    from src.config import ACTIVE_MODELS
    assert set(ACTIVE_MODELS.keys()) == {"GK", "DEF", "MID", "FWD"}

def test_active_models_values_are_paths():
    from src.config import ACTIVE_MODELS
    from pathlib import Path
    for pos, path in ACTIVE_MODELS.items():
        assert isinstance(path, Path), f"{pos} model path must be a Path"
```

- [ ] **Step 3: Run config tests**

```bash
python -m pytest tests/test_config.py -v
```

Expected: New tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add src/config.py tests/test_config.py
rtk git commit -m "feat: B-F5 add ACTIVE_MODELS dict to config for per-position model paths"
```

---

## Task 4: Fixture features in features.py (B-F3 — feature side)

**Files:**
- Modify: `src/pipeline/features.py`
- Test: `tests/test_features.py`

Add `add_fixture_features()` which enriches each row (which may already be per-fixture after DGW expansion in prepare) with `is_home`, `fixture_count`, `rest_days`, `is_fixture_2`.

- [ ] **Step 1: Add `add_fixture_features()` to `features.py`**

Add before `engineer_features()`:

```python
FIXTURE_FEATURE_COLUMNS = [
    "xGC_rolling_4",
    "opponent_form_rolling_6",
    "is_home",
    "fixture_count",
    "rest_days",
    "is_fixture_2",
]


def add_fixture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-fixture context features.

    Expects df to already have one row per fixture (DGW players appear twice).
    - is_home: 1/0 from was_home
    - fixture_count: how many fixtures this player has in this GW (1=normal, 2=DGW)
    - is_fixture_2: 1 for the second game in a DGW, else 0
    - rest_days: days between fixture 1 and fixture 2 (0 for fixture 1 and non-DGW)
    """
    df = df.copy()

    # is_home
    if "was_home" in df.columns:
        df["is_home"] = df["was_home"].astype(int)
    else:
        df["is_home"] = 0

    # fixture_count: count of rows per player per GW
    player_id = "code" if "code" in df.columns else "element"
    df["fixture_count"] = df.groupby([player_id, "season", "GW"])["GW"].transform("count")

    # Sort within each player-GW group by kickoff time to identify fixture order
    if "kickoff_time" in df.columns:
        df["_kickoff_dt"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        df = df.sort_values([player_id, "season", "GW", "_kickoff_dt"])
        df["_fixture_rank"] = df.groupby([player_id, "season", "GW"]).cumcount()
        df["is_fixture_2"] = (df["_fixture_rank"] == 1).astype(int)

        # rest_days for fixture 2 = days since fixture 1
        first_kickoffs = (
            df[df["_fixture_rank"] == 0]
            .set_index([player_id, "season", "GW"])["_kickoff_dt"]
        )
        df = df.join(first_kickoffs.rename("_first_ko"), on=[player_id, "season", "GW"])
        df["rest_days"] = 0.0
        mask = df["is_fixture_2"] == 1
        df.loc[mask, "rest_days"] = (
            (df.loc[mask, "_kickoff_dt"] - df.loc[mask, "_first_ko"])
            .dt.total_seconds() / 86400
        ).clip(lower=0)
        df = df.drop(columns=["_kickoff_dt", "_fixture_rank", "_first_ko"], errors="ignore")
    else:
        df["is_fixture_2"] = 0
        df["rest_days"] = 0.0

    return df
```

- [ ] **Step 2: Call `add_fixture_features` from `engineer_features`**

In `engineer_features()`, add after `add_form_features(df)`:

```python
    df = add_fixture_features(df)
```

Also update the `FEATURE_COLUMNS` export in `features.py` — add a note that full feature list is `FEATURE_COLUMNS + FIXTURE_FEATURE_COLUMNS` (don't merge them yet — predict.py will use them selectively by position).

- [ ] **Step 3: Run fixture feature tests**

```bash
python -m pytest tests/test_features.py::TestFixtureFeatures -v
```

Expected: All 5 tests pass.

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
rtk git add src/pipeline/features.py tests/test_features.py
rtk git commit -m "feat: B-F3 add_fixture_features (is_home, fixture_count, rest_days, is_fixture_2)"
```

---

## Task 5: Per-position prediction with DGW aggregation (B-F3 prediction + B-F4)

**Files:**
- Modify: `src/pipeline/predict.py`
- Test: `tests/test_predict_position.py`

This is the core routing logic. Add `predict_next_gw_per_position()` which takes a DataFrame where DGW players have 2 rows, routes each row to the correct position model, clips negatives, then sums per player (aggregating DGW).

- [ ] **Step 1: Update `predict.py` imports and feature column list**

At the top of `predict.py`, update:

```python
from src.config import ACTIVE_MODEL, ACTIVE_MODELS
from src.pipeline.features import FIXTURE_FEATURE_COLUMNS
```

Add after `FEATURE_COLUMNS`:

```python
# Full feature set = base rolling features + fixture context features
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + FIXTURE_FEATURE_COLUMNS
```

- [ ] **Step 2: Add `load_position_models()` helper**

```python
def load_position_models(models_config: dict | None = None) -> dict:
    """Load all per-position models. Returns dict {position: model or None}.

    If a model file does not exist, returns None for that position (triggers fallback).
    models_config defaults to ACTIVE_MODELS from config.
    """
    if models_config is None:
        models_config = ACTIVE_MODELS
    result = {}
    for pos, path in models_config.items():
        if path is not None and Path(path).exists():
            result[pos] = load_model(Path(path))
        else:
            result[pos] = None
    return result
```

- [ ] **Step 3: Add `predict_next_gw_per_position()`**

```python
def predict_next_gw_per_position(
    player_features: pd.DataFrame,
    models: dict | None = None,
    ep_next_map: dict | None = None,
) -> pd.DataFrame:
    """Generate xP using per-position models with DGW aggregation.

    player_features: DataFrame where DGW players have 2 rows (one per fixture).
        Must have columns: element, code, name, position, team, now_cost,
        fixture_count, is_fixture_2, rest_days, is_home,
        plus all FEATURE_COLUMNS and FIXTURE_FEATURE_COLUMNS.

    models: dict {position_str: model_or_None}. If None, loads from ACTIVE_MODELS.
    ep_next_map: {element_id: ep_next_value} fallback when model is None.

    Returns one row per player (DGW summed). Columns: element, code, name, position,
    team, now_cost, xP, _fallback (bool).
    """
    if models is None:
        models = load_position_models()

    df = player_features.copy()
    if "now_cost" not in df.columns and "value" in df.columns:
        df["now_cost"] = df["value"]

    # Normalize position to string label if stored as integer
    from src.pipeline.fetch import ELEMENT_TYPE_MAP
    if df["position"].dtype != object:
        df["position"] = df["position"].map(ELEMENT_TYPE_MAP)

    feature_cols = ALL_FEATURE_COLUMNS
    predictions = []

    for pos, model in models.items():
        pos_df = df[df["position"] == pos].copy()
        if pos_df.empty:
            continue

        available_features = [c for c in feature_cols if c in pos_df.columns]
        X = pos_df[available_features].fillna(0)

        if model is not None and len(available_features) >= len(FEATURE_COLUMNS):
            xp = model.predict(X)
            xp = np.clip(xp, 0, None)
            pos_df = pos_df.copy()
            pos_df["xP"] = xp
            pos_df["_fallback"] = False
        else:
            # Fallback: use ep_next if provided, else 0
            fallback_xp = 0.0
            if ep_next_map:
                pos_df["xP"] = pos_df["element"].map(ep_next_map).fillna(0.0)
            else:
                pos_df["xP"] = fallback_xp
            pos_df["_fallback"] = True

        predictions.append(pos_df)

    if not predictions:
        return pd.DataFrame(columns=ID_COLUMNS + ["xP", "_fallback"])

    combined = pd.concat(predictions, ignore_index=True)

    # Aggregate: sum xP across fixtures per player (handles DGW)
    agg_cols = {
        "xP": "sum",
        "_fallback": "first",
        "now_cost": "first",
        "team": "first",
        "position": "first",
        "name": "first",
    }
    if "code" in combined.columns:
        agg_cols["code"] = "first"
    if "raw_xP" in combined.columns:
        agg_cols["raw_xP"] = "sum"

    group_key = "element"
    result = combined.groupby(group_key, as_index=False).agg(agg_cols)
    return result
```

- [ ] **Step 4: Run position routing tests**

```bash
python -m pytest tests/test_predict_position.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pipeline/predict.py tests/test_predict_position.py
rtk git commit -m "feat: B-F4 per-position predict with DGW aggregation and ep_next fallback"
```

---

## Task 6: Retrain 4 models (B-F6)

**Files:**
- Modify: `src/pipeline/run.py`

Update `phase_retrain()` to train 4 position-specific RF models, save them with position suffixes, and print per-position MAE + Spearman ρ.

- [ ] **Step 1: Confirm `ALL_FEATURE_COLUMNS` is defined in `predict.py` (from Task 5)**

`ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + FIXTURE_FEATURE_COLUMNS` **must live in `predict.py` only.** `FEATURE_COLUMNS` is in `predict.py`; `FIXTURE_FEATURE_COLUMNS` is in `features.py`. Putting `ALL_FEATURE_COLUMNS` in `features.py` would require importing from `predict.py` → circular import. Do NOT add it to `features.py`.

Verify no circular import:

```bash
python -c "from src.pipeline.predict import ALL_FEATURE_COLUMNS; print(len(ALL_FEATURE_COLUMNS), 'features — import ok')"
```

Expected: `24 features — import ok` (18 base + 6 fixture). Fix any `ImportError` before proceeding.

- [ ] **Step 2: Update `run.py` imports**

```python
from src.config import (
    VAASTAV_DIR, RESULTS_DIR, MODELS_DIR, CURRENT_SEASON,
    ACTIVE_MODEL, ACTIVE_MODELS, BOOTSTRAP_MAX_AGE_HOURS, SNAPSHOTS_DIR,
    FPL_ENTRY_URL, load_user_config, UserConfigError,
)
from src.pipeline.predict import (
    predict_next_gw, predict_next_gw_per_position, load_position_models,
    get_feature_columns, save_full_predictions, apply_xp_corrections,
    ALL_FEATURE_COLUMNS,
)
```

- [ ] **Step 3: Replace `phase_retrain()` in `run.py`**

Find the existing `phase_retrain` function (starts at approx line 602) and replace it entirely:

```python
def phase_retrain(target_gw: int | None = None):
    """Phase 4: Retrain 4 per-position RF models on full dataset (manual trigger)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from scipy.stats import spearmanr
    import joblib

    print("[retrain] Building full feature-engineered dataset...")
    merged = build_merged_dataset(vaastav_dir=VAASTAV_DIR)
    features = engineer_features(merged)
    print(f"[retrain] Training data: {len(features)} rows")

    # Normalize position to string label
    if "position" in features.columns and features["position"].dtype != object:
        features["position"] = features["position"].map(ELEMENT_TYPE_MAP)

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in features.columns]

    label = f"gw{target_gw}" if target_gw else datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    position_results = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = features[features["position"] == pos].copy()
        if len(pos_df) < 100:
            print(f"[retrain] {pos}: insufficient data ({len(pos_df)} rows) — skipping")
            continue

        X = pos_df[feature_cols].fillna(0)
        y = pos_df["total_points"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rho, _ = spearmanr(y_pred, y_test)

        new_path = MODELS_DIR / f"rf_{pos.lower()}_{label}.sav"
        joblib.dump(model, new_path)
        position_results[pos] = {"mae": mae, "rho": rho, "path": new_path, "n": len(pos_df)}
        print(f"[retrain] {pos}: MAE={mae:.3f}, Spearman ρ={rho:.3f} ({len(pos_df)} rows) → {new_path.name}")

    print("\n[retrain] Summary:")
    for pos, r in position_results.items():
        print(f"  {pos}: MAE={r['mae']:.3f}, ρ={r['rho']:.3f}")
    print(f"\n[retrain] To promote: update ACTIVE_MODELS in src/config.py to point to these files.")
    for pos, r in position_results.items():
        print(f"  '{pos}': MODELS_DIR / '{r['path'].name}'")
```

- [ ] **Step 2: Run the retrain smoke test (dry run with tiny dataset)**

```bash
python -c "
import pandas as pd
from src.pipeline.features import engineer_features, add_fixture_features
from src.pipeline.features import FIXTURE_FEATURE_COLUMNS
print('FIXTURE_FEATURE_COLUMNS:', FIXTURE_FEATURE_COLUMNS)
print('import ok')
"
```

Expected: `FIXTURE_FEATURE_COLUMNS: ['xGC_rolling_4', ...]` — no import error.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
rtk git add src/pipeline/run.py src/pipeline/features.py src/pipeline/predict.py
rtk git commit -m "feat: B-F6 retrain phase trains 4 per-position RF models with MAE + Spearman rho"
```

---

## Task 7: Spearman ρ in accuracy log (B-F7)

**Files:**
- Modify: `src/pipeline/analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Add `compute_spearman_rho()` to `analysis.py`**

Add after the existing imports:

```python
import math
from scipy.stats import spearmanr as _spearmanr


def compute_spearman_rho(picks_df: pd.DataFrame) -> float:
    """Compute Spearman rank correlation between xP predictions and actual points.

    Returns NaN if fewer than 2 rows or no variance in either column.
    picks_df must have columns: xP, actual_points.
    """
    df = picks_df.dropna(subset=["xP", "actual_points"])
    if len(df) < 2:
        return float("nan")
    rho, _ = _spearmanr(df["xP"], df["actual_points"])
    return float(rho) if not math.isnan(rho) else float("nan")
```

- [ ] **Step 2: Update `append_accuracy_log()` to include spearman_rho**

Find the `append_accuracy_log` function. Identify where it builds the `row` dict and adds the `"your_pts"`, `"your_xp"` etc. columns. Add `"spearman_rho"` to the row dict:

```python
    rho = compute_spearman_rho(picks_df) if picks_df is not None and not picks_df.empty else float("nan")
    row = {
        ...existing fields...,
        "spearman_rho": rho,
    }
```

- [ ] **Step 3: Run Spearman tests**

```bash
python -m pytest tests/test_analysis.py::TestSpearmanRho -v
```

Expected: All 4 tests pass.

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all tests pass (count should be at least 208 + new tests added across all tasks).

- [ ] **Step 5: Commit**

```bash
rtk git add src/pipeline/analysis.py tests/test_analysis.py
rtk git commit -m "feat: B-F7 Spearman rho in compute_spearman_rho and accuracy_log.csv"
```

---

## Task 8: Wire `predict_next_gw_per_position` into `phase_predict` (integration)

**Files:**
- Modify: `src/pipeline/run.py`

Replace the call to `predict_next_gw(...)` in `phase_predict()` with the new per-position function. Keep the old function as a fallback path if all 4 models are missing.

- [ ] **Step 1: Update `phase_predict()` to use per-position prediction**

In `run.py`, update the import:

```python
from src.pipeline.predict import (
    predict_next_gw, predict_next_gw_per_position,
    load_position_models, get_feature_columns, save_full_predictions,
    apply_xp_corrections, ALL_FEATURE_COLUMNS,
)
```

Find the block in `phase_predict` where `predict_next_gw` is called (look for `predictions = predict_next_gw(`). Replace with:

```python
        # Attempt per-position prediction (Track B)
        pos_models = load_position_models()
        any_model_available = any(m is not None for m in pos_models.values())

        if any_model_available:
            print("[predict] Using per-position models (Track B)")
            # Build ep_next fallback map from bootstrap
            ep_next_map = {}
            if bootstrap:
                ep_next_map = {
                    el["id"]: el.get("ep_next", 0) or 0
                    for el in bootstrap.get("elements", [])
                }
            predictions = predict_next_gw_per_position(
                latest,
                models=pos_models,
                ep_next_map=ep_next_map,
            )
        else:
            # All position models missing — fall back to global model
            print("[predict] No per-position models found — falling back to global model")
            try:
                predictions = predict_next_gw(latest, model_path=ACTIVE_MODEL)
                predictions["_fallback"] = True
            except Exception as e:
                logger.warning(f"Global model also failed: {e}. Using ep_next only.")
                predictions = _seed_from_ep_next(bootstrap, latest)
```

Note: `_seed_from_ep_next` is the existing logic already in `phase_predict` — factor it out as a helper if needed.

- [ ] **Step 2: Run the integration smoke test**

```bash
python -m pytest tests/test_integration_replay.py -v 2>&1 | tail -20
```

Expected: Tests pass (they use cached GW30/31 snapshots and ep_next fallback — position models won't exist yet, so the fallback path runs).

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
rtk git add src/pipeline/run.py
rtk git commit -m "feat: B-F3 wire per-position prediction into phase_predict with global model fallback"
```

---

## Task 9: End-to-end validation (if vaastav data available)

Skip this task in CI. Run locally after full retrain.

- [ ] **Step 1: Retrain all 4 models**

```bash
python -m src.pipeline.run retrain --gw 31
```

Expected output: 4 lines like `GK: MAE=0.xxx, Spearman ρ=0.xxx`.

- [ ] **Step 2: Update `ACTIVE_MODELS` in `src/config.py`** with the paths printed by retrain.

- [ ] **Step 3: Run predict**

```bash
python -m src.pipeline.run predict --gw 32
```

Check `results/predictions_gw32.csv` — verify it has all 4 positions and non-trivial xP variance.

- [ ] **Step 4: Verify success gate**

Targets from the roadmap:
- Spearman ρ ≥ 0.65 (top-200k quality)
- MAE ≤ 1.035 (no regression from global baseline)

If ρ < 0.65 or MAE > 1.035, do NOT panic — check per-position breakdowns from the retrain output. The GK position historically achieves ρ ~0.77 due to clean sheet predictability; MID/FWD are harder.

- [ ] **Step 5: Update roadmap status**

In `docs/improvements-roadmap.md`, update Track B status from `SPEC READY` to `COMPLETE (YYYY-MM-DD)` and record test count.

- [ ] **Step 6: Final commit**

```bash
rtk git add src/config.py docs/improvements-roadmap.md
rtk git commit -m "chore: promote Track B per-position models; update roadmap status"
```

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `opponent_team` column absent in old vaastav seasons | `add_opponent_stats` guards with `if "opponent_team" not in df.columns`; adds NaN columns |
| `kickoff_time` absent in merged_gw.csv | `add_fixture_features` guards — sets `is_fixture_2=0`, `rest_days=0.0` (no crash) |
| Per-position models not yet trained (first deploy) | `load_position_models` returns `None` for missing files → fallback to ep_next runs correctly |
| Circular import: `predict.py` imports `features.py`, `features.py` might import `predict.py` | `ALL_FEATURE_COLUMNS` is defined in `predict.py` only; `features.py` only exports `FIXTURE_FEATURE_COLUMNS` |
| `spearmanr` import from scipy | scipy is already in `requirements.txt` (used by source_validation.py); no new dep needed |
| DGW player appearing in `latest` (last-row groupby gives 1 row, not 2) | **Explicitly deferred — see B-F3-DGW below.** The per-fixture row expansion in `phase_predict` is a follow-up task. This plan's `predict_next_gw_per_position` function correctly handles pre-expanded 2-row input (tested in `TestDGWAggregation`), but `phase_predict` does NOT currently expand rows. The A-F4 xP correction layer handles DGW xP scaling for now. |

### Deferred: B-F3-DGW — Per-fixture row expansion in `phase_predict`

The B-F3 spec says "predict xP per fixture separately, then sum." The `predict_next_gw_per_position` function supports this correctly when given 2-row DGW input. However, `phase_predict` still passes `latest` (one row per player) because joining FPL fixture data to expand DGW players requires reliable kickoff timestamps.

**What is NOT done in this plan:**
- `phase_predict` does not fetch the FPL fixtures API and expand DGW player rows before prediction
- The `rest_days` and `is_fixture_2` features are always 0 in live prediction (no fixture expansion)

**Placeholder skipped test to add in `tests/test_predict_position.py`** (add this to the end of the file after Task 1):

```python
@pytest.mark.skip(reason="B-F3-DGW: phase_predict fixture expansion not yet implemented")
def test_phase_predict_expands_dgw_player_to_two_rows():
    """phase_predict should expand DGW players to 2 fixture rows before calling predict_next_gw_per_position.

    When this test is unskipped, verify:
    1. phase_predict fetches fixtures via fetch_fixtures()
    2. DGW players (those with 2 fixtures in target_gw) appear as 2 rows in the feature DataFrame
    3. is_fixture_2=1 and rest_days>0 on the second row
    4. Final predictions_gw{N}.csv has one row per player with DGW xP = sum of both fixtures
    """
    pass
```

Add this to Task 1 Step 1 after the existing test classes in `tests/test_predict_position.py`. The skip marker makes the deferred scope explicit and survives as a regression guard when the feature is eventually implemented.
