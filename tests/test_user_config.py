import pytest
import tempfile
from pathlib import Path
from src.config import load_user_config, UserConfigError


class TestLoadUserConfig:
    def test_loads_valid_config(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 1681779
    label: "Main"
preferences:
  horizon_gws: 5
  max_hit_points: 8
  fdr_sensitivity: 0.15
""")
        cfg = load_user_config(cfg_file)
        assert cfg["teams"]["default"]["entry_id"] == 1681779
        assert cfg["preferences"]["horizon_gws"] == 5
        assert cfg["preferences"]["fdr_sensitivity"] == 0.15

    def test_raises_when_file_missing(self, tmp_path):
        with pytest.raises(UserConfigError, match="user_config.yaml"):
            load_user_config(tmp_path / "nonexistent.yaml")

    def test_raises_when_entry_id_missing(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    label: Main\n")
        with pytest.raises(UserConfigError, match="entry_id"):
            load_user_config(cfg_file)

    def test_raises_when_entry_id_not_int(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    entry_id: abc\n")
        with pytest.raises(UserConfigError, match="entry_id"):
            load_user_config(cfg_file)

    def test_raises_when_horizon_out_of_range(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 123
preferences:
  horizon_gws: 10
""")
        with pytest.raises(UserConfigError, match="horizon_gws"):
            load_user_config(cfg_file)

    def test_defaults_applied_when_preferences_missing(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("teams:\n  default:\n    entry_id: 123\n")
        cfg = load_user_config(cfg_file)
        assert cfg["preferences"]["horizon_gws"] == 5
        assert cfg["preferences"]["max_hit_points"] == 8
        assert cfg["preferences"]["fdr_sensitivity"] == 0.15

    def test_alt_team_optional(self, tmp_path):
        cfg_file = tmp_path / "user_config.yaml"
        cfg_file.write_text("""
teams:
  default:
    entry_id: 123
  alt:
    entry_id: 456
    label: Experimental
""")
        cfg = load_user_config(cfg_file)
        assert cfg["teams"]["alt"]["entry_id"] == 456
