"""Future-fit isolation, selection identity, dates, exports, and artifacts."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.evaluation.experiment import run_evaluation
from src.demand_planning.forecasting.artifacts import forecast_csv_bytes, save_forecast
from src.demand_planning.forecasting.future import forecast_selected_model, select_forecast_horizon
from src.demand_planning.forecasting.models import MODEL_NAMES, FittedModel, predict_recursive

CONTEXT = {"source": "Future test fixture", "provenance": "Synthetic", "store": 1, "family": "A"}


def history() -> pd.Series:
    """Known complete history crossing a leap-year boundary."""
    return pd.Series(20. + np.arange(146) * .03 + 3 * np.sin(np.arange(146) * 2 * np.pi / 7),
                     index=pd.date_range("2023-12-01", periods=146))


@pytest.fixture(scope="module")
def evaluation():
    """Real chronological evaluation; future fitting must not alter its evidence."""
    return run_evaluation(history(), **CONTEXT)


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_future_refit_preserves_holdout_evidence_and_dates(evaluation, horizon: int) -> None:
    metadata = deepcopy(evaluation.metadata)
    test_scores = evaluation.test_metrics.copy(deep=True)
    predictions = evaluation.predictions.copy(deep=True)
    prior_model_hash = joblib.hash(evaluation.selected_model)
    result = forecast_selected_model(history(), evaluation, horizon, **CONTEXT)
    assert len(result.forecasts) == horizon
    assert result.forecasts.date.tolist() == list(pd.date_range(history().index[-1], periods=horizon + 1)[1:])
    assert result.model.trained_through == history().index[-1]
    assert result.model.trained_through > evaluation.selected_model.trained_through
    assert result.model is not evaluation.selected_model
    assert result.metadata["evaluation_trained_through"] == metadata["trained_through"]
    assert result.forecasts.forecast_origin.eq(history().index[-1]).all()
    assert result.forecasts.forecast.ge(0).all()
    assert np.isfinite(result.forecasts.forecast).all()
    assert evaluation.metadata == metadata
    pd.testing.assert_frame_equal(evaluation.test_metrics, test_scores)
    pd.testing.assert_frame_equal(evaluation.predictions, predictions)
    assert joblib.hash(evaluation.selected_model) == prior_model_hash
    assert result.metadata["interval_coverage"] is None
    assert "lower" not in result.forecasts and "upper" not in result.forecasts


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fresh_refit_supports_each_model(evaluation, name: str) -> None:
    """Controlled selection fixture exercises all refit branches, not accuracy claims."""
    chosen = deepcopy(evaluation)
    chosen.validation_metrics["wape_pct"] = 100.
    chosen.validation_metrics.loc[chosen.validation_metrics.model.eq(name), "wape_pct"] = 0.
    chosen.metadata["selected_model"] = name
    # No usable old estimator: the service must create and train a fresh one.
    chosen.selected_model = FittedModel(name, None, (), evaluation.selected_model.trained_through, 0)
    result = forecast_selected_model(history(), chosen, 30, **CONTEXT)
    assert result.model.name == name
    assert result.forecasts.model.eq(name).all()
    assert result.model.training_rows == (146 if name == "Seasonal Naive" else 118)
    if name != "Seasonal Naive":
        assert result.model.estimator is not None
    assert chosen.selected_model.estimator is None


@pytest.mark.parametrize("field,value", [("source", "another.csv"), ("provenance", "Unverified"),
                                        ("store", 2), ("family", "B")])
def test_cross_source_or_series_evaluation_rejected(evaluation, field: str, value: object) -> None:
    context = {**CONTEXT, field: value}
    with pytest.raises(ValueError, match="matching evaluation"):
        forecast_selected_model(history(), evaluation, 7, **context)


def test_changed_data_dates_or_selection_rejected(evaluation) -> None:
    changed = history()
    changed.iloc[-1] += 1
    for invalid in (changed, history().iloc[1:], history().iloc[:-1]):
        with pytest.raises(ValueError, match="matching evaluation"):
            forecast_selected_model(invalid, evaluation, **CONTEXT)
    altered = deepcopy(evaluation)
    altered.metadata["selected_model"] = "Not the winner"
    with pytest.raises(ValueError, match="inconsistent"):
        forecast_selected_model(history(), altered, **CONTEXT)
    altered = deepcopy(evaluation)
    altered.metadata["experiment_version"] = "obsolete"
    with pytest.raises(ValueError, match="matching evaluation"):
        forecast_selected_model(history(), altered, **CONTEXT)


@pytest.mark.parametrize("horizon", [0, -1, 8, True, 14.0, "30", None])
def test_invalid_future_horizons_fail_before_refit(evaluation, horizon: object) -> None:
    with pytest.raises(ValueError, match="horizon"):
        forecast_selected_model(history(), evaluation, horizon, **CONTEXT)


def test_prefixes_do_not_mutate_or_retrain(evaluation) -> None:
    full = forecast_selected_model(history(), evaluation, 30, **CONTEXT)
    prefix = select_forecast_horizon(full, 7)
    assert prefix.model is full.model
    np.testing.assert_array_equal(prefix.forecasts.forecast, full.forecasts.forecast.iloc[:7])
    assert prefix.metadata["horizon_days"] == 7
    assert full.metadata["horizon_days"] == 30
    assert prefix.forecasts.horizon_days.eq(7).all()
    assert full.forecasts.horizon_days.eq(30).all()
    with pytest.raises(ValueError, match="exceeds"):
        select_forecast_horizon(prefix, 30)
    with pytest.raises(ValueError, match="horizon"):
        select_forecast_horizon(full, True)


def test_future_actuals_outside_cutoff_do_not_enter_forecast(evaluation) -> None:
    sales = history()
    frame = pd.DataFrame({"date": pd.date_range(sales.index[0], periods=160),
                          "store_nbr": 1, "family": "A", "sales": 99999.})
    frame.loc[:145, "sales"] = sales.to_numpy()
    supplied = daily_sales(frame, 1, "A", sales.index[0], sales.index[-1])
    first = forecast_selected_model(supplied, evaluation, 7, **CONTEXT)
    frame.loc[146:, "sales"] = 0.
    second = forecast_selected_model(daily_sales(frame, 1, "A", sales.index[0], sales.index[-1]), evaluation, 7, **CONTEXT)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)


def test_csv_and_trusted_artifact_round_trip(evaluation, tmp_path: Path) -> None:
    result = forecast_selected_model(history(), evaluation, 14, **CONTEXT)
    csv = pd.read_csv(BytesIO(forecast_csv_bytes(result.forecasts)))
    assert len(csv) == 14
    assert csv.date.iloc[0] == result.metadata["forecast_start"]
    assert csv.family.eq("A").all() and csv.store_nbr.eq(1).all()
    assert csv.provenance.eq("Synthetic").all()
    assert csv.uncertainty_status.eq("unavailable").all()
    assert csv.forecast_origin.eq(str(history().index[-1].date())).all()
    output = save_forecast(result, tmp_path / "future")
    metadata = json.loads((output / "forecast_metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_scope"] != evaluation.metadata["artifact_scope"]
    assert metadata["trained_through"] != evaluation.metadata["trained_through"]
    # This is a trusted file just created by this test, not an uploaded artifact.
    bundle = joblib.load(output / "future_model.joblib")
    restored = predict_recursive(bundle["model"], bundle["history_tail"], 14)
    np.testing.assert_allclose(restored.forecast, result.forecasts.forecast)
    with pytest.raises(FileExistsError):
        save_forecast(result, output)


def test_zero_demand_future_stays_zero() -> None:
    sales = pd.Series(0., index=pd.date_range("2024-01-01", periods=116))
    evaluated = run_evaluation(sales, **CONTEXT)
    result = forecast_selected_model(sales, evaluated, 30, **CONTEXT)
    assert result.forecasts.forecast.eq(0).all()
