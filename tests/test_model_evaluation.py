"""Metric arithmetic, real estimator integration, and holdout isolation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.demand_planning.evaluation.artifacts import save_evaluation
from src.demand_planning.evaluation.experiment import run_evaluation, select_model
from src.demand_planning.evaluation.metrics import forecast_metrics
from src.demand_planning.evaluation.splits import chronological_split
from src.demand_planning.features.calendar_lags import build_features
from src.demand_planning.forecasting.models import MODEL_NAMES, FittedModel, fit_model, predict_recursive


def demand(days: int = 146) -> pd.Series:
    """Small, deterministic daily series with one available diagnostic CV fold."""
    dates = pd.date_range("2024-01-01", periods=days)
    return pd.Series(20 + np.arange(days) * .02 + 3 * np.sin(np.arange(days) * 2 * np.pi / 7), index=dates)


@pytest.fixture(scope="module")
def result():
    """Run actual sklearn/XGBoost estimators once for contract assertions."""
    return run_evaluation(demand(), "Test fixture", "Synthetic", 1, "A")


def test_metrics_hand_calculation_and_zero_cases() -> None:
    metrics = forecast_metrics([0, 2, 4], [1, 1, 6])
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2))
    assert metrics["wape_pct"] == pytest.approx(100 * 4 / 6)
    assert forecast_metrics([0, 0], [0, 0]) == {"mae": 0., "rmse": 0., "wape_pct": None}
    assert forecast_metrics([0, 0], [2, 4]) == {"mae": 3., "rmse": np.sqrt(10), "wape_pct": None}
    assert forecast_metrics([0, 1], [-9, 1])["mae"] == 0


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([[1]], [[1]]),
                                           ([np.nan], [0]), ([0], [np.inf]), ([-1], [0])])
def test_invalid_metrics_rejected(actual, predicted) -> None:
    with pytest.raises(ValueError):
        forecast_metrics(actual, predicted)


def test_same_origins_horizons_and_winner_only_test(result) -> None:
    assert set(result.validation_metrics.model) == set(MODEL_NAMES)
    assert len(result.validation_metrics) == 12
    assert result.validation_metrics.origin.nunique() == 1
    assert set(result.validation_metrics.horizon) == {7, 14, 30}
    assert result.cv_metrics.fold.nunique() == 1
    assert result.cv_metrics.origin.max() < result.validation_metrics.origin.min()
    assert set(result.test_metrics.model) == {result.selected_model.name}
    assert len(result.test_metrics) == 3
    assert result.selected_model.trained_through == chronological_split(demand()).validation.index[-1]
    assert result.predictions.forecast.ge(0).all()
    assert np.isfinite(result.predictions.forecast).all()
    for _, group in result.predictions.groupby(["stage", "fold"]):
        assert group.groupby("model").date.apply(tuple).nunique() == 1
    json.dumps(result.metadata, allow_nan=False)


def test_future_test_actuals_do_not_change_selection_or_predictions(result) -> None:
    modified = demand()
    modified.iloc[-30:] = 100000
    changed = run_evaluation(modified)
    pd.testing.assert_frame_equal(result.validation_metrics, changed.validation_metrics)
    pd.testing.assert_frame_equal(result.cv_metrics, changed.cv_metrics)
    assert result.selected_model.name == changed.selected_model.name
    pd.testing.assert_frame_equal(result.predictions.drop(columns="actual"), changed.predictions.drop(columns="actual"))
    assert not result.test_metrics.equals(changed.test_metrics)


def test_scaling_fits_only_training_examples() -> None:
    train = chronological_split(demand()).train
    fitted = fit_model("Linear Regression", train)
    expected = build_features(train).dropna().mean().to_numpy()
    np.testing.assert_allclose(fitted.estimator.named_steps["standardscaler"].mean_, expected)
    assert fitted.training_rows == len(train) - 28


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_seeded_estimators_are_reproducible(name: str) -> None:
    history = demand(56)
    first, second = fit_model(name, history), fit_model(name, history)
    pd.testing.assert_frame_equal(predict_recursive(first, history, 7), predict_recursive(second, history, 7))


def test_zero_demand_fallback_and_tie_break() -> None:
    result = run_evaluation(pd.Series(0., index=pd.date_range("2024-01-01", periods=116)))
    assert result.cv_metrics.empty
    assert result.validation_metrics.wape_pct.isna().all()
    assert result.selected_model.name == "Seasonal Naive"
    assert result.test_metrics.mae.eq(0).all()
    scores = result.validation_metrics.copy()
    scores.loc[scores.model.eq("Seasonal Naive"), "mae"] = 1
    assert select_model(scores) == "Linear Regression"
    scores["wape_pct"] = 3.
    scores.loc[scores.model.eq("XGBoost"), "wape_pct"] = 1.
    assert select_model(scores) == "XGBoost"


def test_artifact_round_trip_and_overwrite_protection(result, tmp_path: Path) -> None:
    directory = save_evaluation(result, tmp_path / "run")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_model"] == result.selected_model.name
    # Only load the trusted artifact just created within this test.
    bundle = joblib.load(directory / "selected_model.joblib")
    forecast = predict_recursive(bundle["model"], bundle["history_tail"], 30)
    expected = result.predictions.loc[result.predictions.stage.eq("test")].forecast
    np.testing.assert_allclose(forecast.forecast, expected)
    assert len(pd.read_csv(directory / "selected_test_metrics.csv")) == 3
    with pytest.raises(FileExistsError):
        save_evaluation(result, directory)


class PreviousPrediction:
    """Test estimator that exposes recursive feedback through lag_1."""

    def predict(self, features):
        return features.lag_1.to_numpy() - 2


def test_recursive_feedback_clips_before_next_step() -> None:
    history = pd.Series(3., index=pd.date_range("2024-01-01", periods=56))
    fitted = FittedModel("Test estimator", PreviousPrediction(), tuple(build_features(history).columns), history.index[-1], 28)
    forecast = predict_recursive(fitted, history, 7)
    assert forecast.forecast.tolist() == [1., 0., 0., 0., 0., 0., 0.]
    with pytest.raises(ValueError, match="trained after"):
        predict_recursive(fitted, history.iloc[:-1], 7)
