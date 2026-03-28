import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from src.pipeline.prepare import (
    load_season_gw_data,
    load_live_gw_files,
    merge_seasons,
    add_fixture_difficulty,
    build_merged_dataset,
)


class TestLoadSeasonGwData:
    def test_loads_csv_with_season_column(self, tmp_path):
        gw_dir = tmp_path / "data" / "Fantasy-Premier-League" / "data" / "2024-25" / "gws"
        gw_dir.mkdir(parents=True)
        df = pd.DataFrame({"name": ["Saka"], "total_points": [8], "GW": [1]})
        df.to_csv(gw_dir / "merged_gw.csv", index=False)

        result = load_season_gw_data("2024-25", vaastav_dir=tmp_path / "data" / "Fantasy-Premier-League")
        assert "season" in result.columns
        assert result["season"].iloc[0] == "2024-25"

    def test_handles_latin1_encoding(self, tmp_path):
        gw_dir = tmp_path / "data" / "Fantasy-Premier-League" / "data" / "2017-18" / "gws"
        gw_dir.mkdir(parents=True)
        df = pd.DataFrame({"name": ["Agüero"], "total_points": [10], "GW": [1]})
        df.to_csv(gw_dir / "merged_gw.csv", index=False, encoding="latin-1")

        result = load_season_gw_data("2017-18", vaastav_dir=tmp_path / "data" / "Fantasy-Premier-League")
        assert result["name"].iloc[0] == "Agüero"


class TestLoadLiveGwFiles:
    def test_discovers_live_csv_files(self, tmp_path):
        gw_dir = tmp_path / "gws"
        gw_dir.mkdir()
        df = pd.DataFrame({"name": ["Saka"], "total_points": [8], "GW": [30], "element": [3]})
        df.to_csv(gw_dir / "gw30_live.csv", index=False)
        df2 = pd.DataFrame({"name": ["Saka"], "total_points": [6], "GW": [31], "element": [3]})
        df2.to_csv(gw_dir / "gw31_live.csv", index=False)

        result = load_live_gw_files(gw_dir)
        assert len(result) == 2

    def test_returns_empty_when_no_live_files(self, tmp_path):
        gw_dir = tmp_path / "gws"
        gw_dir.mkdir()
        result = load_live_gw_files(gw_dir)
        assert len(result) == 0

    def test_dedup_prefers_vaastav_over_live(self, tmp_path):
        """When vaastav merged_gw.csv covers a GW, live data is dropped."""
        gw_dir = tmp_path / "data" / "FPL" / "data" / "2025-26" / "gws"
        gw_dir.mkdir(parents=True)

        # vaastav data covers GW30
        vaastav_df = pd.DataFrame({
            "name": ["Saka"], "total_points": [8], "GW": [30],
            "element": [3], "tackles": [2],  # richer columns
        })
        vaastav_df.to_csv(gw_dir / "merged_gw.csv", index=False)

        # live data also has GW30
        live_df = pd.DataFrame({
            "name": ["Saka"], "total_points": [8], "GW": [30], "element": [3],
        })
        live_df.to_csv(gw_dir / "gw30_live.csv", index=False)

        # live GW31 not in vaastav
        live_df2 = pd.DataFrame({
            "name": ["Saka"], "total_points": [6], "GW": [31], "element": [3],
        })
        live_df2.to_csv(gw_dir / "gw31_live.csv", index=False)

        result = build_merged_dataset(
            seasons=["2025-26"],
            vaastav_dir=tmp_path / "data" / "FPL",
        )
        # Should have GW30 from vaastav (with tackles) and GW31 from live
        assert len(result) == 2
        gw30 = result[result["GW"] == 30]
        assert "tackles" in result.columns
        assert gw30.iloc[0]["tackles"] == 2  # vaastav row, not live


class TestMergeSeasons:
    def test_concatenates_with_common_columns(self):
        df1 = pd.DataFrame({"name": ["A"], "total_points": [5], "season": ["2023-24"]})
        df2 = pd.DataFrame({"name": ["B"], "total_points": [3], "season": ["2024-25"], "tackles": [2]})
        result = merge_seasons([df1, df2])
        assert len(result) == 2
        assert "tackles" in result.columns  # schema union, not intersection


class TestAddFixtureDifficulty:
    def test_adds_fdr_columns(self, tmp_path):
        fixtures_path = tmp_path / "fixtures.csv"
        fixtures = pd.DataFrame({
            "id": [1], "event": [1], "team_h": [1], "team_a": [10],
            "team_h_difficulty": [3], "team_a_difficulty": [4],
        })
        fixtures.to_csv(fixtures_path, index=False)

        gw_df = pd.DataFrame({
            "fixture": [1], "was_home": [True], "season": ["2025-26"],
        })
        result = add_fixture_difficulty(gw_df, fixtures_path)
        assert "fdr_team" in result.columns
        assert result["fdr_team"].iloc[0] == 3


class TestBuildMergedDataset:
    def test_end_to_end_produces_expected_columns(self, tmp_path):
        season_dir = tmp_path / "FPL" / "data" / "2025-26"
        gw_dir = season_dir / "gws"
        gw_dir.mkdir(parents=True)

        gw_data = pd.DataFrame({
            "name": ["Saka", "Saka"], "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"], "element": [3, 3],
            "total_points": [8, 6], "minutes": [90, 90],
            "fixture": [1, 2], "was_home": [True, False],
            "GW": [1, 2], "xP": [6.5, 5.0],
            "goals_scored": [1, 0], "assists": [1, 0],
            "clean_sheets": [0, 0], "ict_index": [12.0, 5.0],
            "influence": [40.0, 15.0], "creativity": [35.0, 10.0],
            "threat": [50.0, 20.0], "bps": [35, 12], "bonus": [3, 0],
            "value": [105, 105], "transfers_in": [5000, 3000],
            "transfers_out": [1000, 2000], "selected": [3000000, 3100000],
            "opponent_team": [10, 15], "round": [1, 2],
        })
        gw_data.to_csv(gw_dir / "merged_gw.csv", index=False)

        fixtures = pd.DataFrame({
            "id": [1, 2], "event": [1, 2],
            "team_h": [1, 10], "team_a": [10, 1],
            "team_h_difficulty": [3, 4], "team_a_difficulty": [4, 3],
        })
        fixtures.to_csv(season_dir / "fixtures.csv", index=False)

        result = build_merged_dataset(
            seasons=["2025-26"],
            vaastav_dir=tmp_path / "FPL",
        )
        assert len(result) == 2
        assert "season" in result.columns
        assert "fdr_team" in result.columns
