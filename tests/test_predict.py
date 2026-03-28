# tests/test_predict.py
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.pipeline.predict import (
    load_model,
    get_feature_columns,
    predict_next_gw,
)


@pytest.fixture
def trained_model(tmp_path):
    """Create a mock model .sav file."""
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model.fit(X, y)
    model_path = tmp_path / "rf_model.sav"
    joblib.dump(model, model_path)
    return model_path, model.feature_names_in_ if hasattr(model, "feature_names_in_") else None


class TestLoadModel:
    def test_loads_saved_model(self, trained_model):
        model_path, _ = trained_model
        model = load_model(model_path)
        assert hasattr(model, "predict")

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.sav")


class TestGetFeatureColumns:
    def test_returns_expected_feature_set(self):
        cols = get_feature_columns()
        assert "total_points_roll_4" in cols
        assert "minutes_roll_4" in cols
        assert "transfers_net" in cols
        # Should NOT include target or identifiers
        assert "total_points" not in cols
        assert "name" not in cols
        assert "element" not in cols


class TestPredictNextGW:
    def test_returns_dataframe_with_xp_column(self, trained_model):
        model_path, _ = trained_model
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        # Create model that expects named features
        feature_cols = get_feature_columns()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(50, len(feature_cols)), columns=feature_cols)
        y = np.random.rand(50)
        model.fit(X, y)
        named_model_path = model_path.parent / "rf_named.sav"
        joblib.dump(model, named_model_path)

        player_features = X.copy()
        player_features["element"] = range(50)
        player_features["name"] = [f"Player_{i}" for i in range(50)]
        player_features["position"] = ["MID"] * 50
        player_features["team"] = ["Arsenal"] * 50
        player_features["now_cost"] = [100] * 50

        result = predict_next_gw(player_features, named_model_path)
        assert "xP" in result.columns
        assert "element" in result.columns
        assert len(result) == 50
        assert (result["xP"] >= 0).all()


class TestSaveFullPredictionsCSV:
    def test_save_predictions_creates_file(self, sample_predictions_df, tmp_path):
        from src.pipeline.predict import save_full_predictions
        out_path = tmp_path / "predictions_gw33.csv"
        save_full_predictions(sample_predictions_df, out_path)
        assert out_path.exists()
        import pandas as pd
        df = pd.read_csv(out_path)
        assert list(df.columns) == ["element", "code", "name", "position", "team", "xP", "now_cost"]
        assert len(df) == len(sample_predictions_df)

    def test_save_predictions_cost_in_01m_units(self, sample_predictions_df, tmp_path):
        from src.pipeline.predict import save_full_predictions
        # now_cost stays in 0.1M units (FPL convention): 105 = £10.5m stored as 105
        out_path = tmp_path / "predictions_gw33.csv"
        save_full_predictions(sample_predictions_df, out_path)
        import pandas as pd
        df = pd.read_csv(out_path)
        # Saka: now_cost=105 stays as 105 (0.1M units)
        saka = df[df["name"] == "Saka"].iloc[0]
        assert saka["now_cost"] == 105
