"""Separate full-history refits from the immutable holdout assessment evidence."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from src.demand_planning.data.time_series import validate_daily_series
from src.demand_planning.evaluation.experiment import EXPERIMENT_VERSION, EvaluationResult, select_model, series_fingerprint
from src.demand_planning.forecasting.models import FittedModel, fit_model, predict_recursive

FORECAST_VERSION = "phase5-v1"
INTERVAL_REASON = (
    "Intervals unavailable: evaluation supplies at most three earlier origins per lead time, "
    "no separate interval calibration sample, and no independent interval coverage assessment. "
    "Selection-validation and test errors are not repurposed as calibration data."
)


@dataclass
class FutureForecast:
    """Fresh fit, future rows, and provenance; original evaluation stays unchanged."""

    model: FittedModel
    forecasts: pd.DataFrame
    history_tail: pd.Series
    metadata: dict


def validate_evaluation_context(
    history: pd.Series, evaluation: EvaluationResult, *, source: str, provenance: str,
    store: int | None, family: str | None,
) -> None:
    """Reject stale/cross-series selection and mismatched evaluation versions.

    Exact date/value matching is deliberate: added or changed observations need
    an explicit new evaluation, not silent reuse of an unrelated selection.
    """
    expected = {"experiment_version": EXPERIMENT_VERSION, "data_fingerprint": series_fingerprint(history),
                "source": source, "provenance": provenance, "store": store, "family": family}
    if any(evaluation.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("No matching evaluation for these data and filters. Run Model Evaluation for the current source, store, family, and dates first.")
    winner = select_model(evaluation.validation_metrics)
    if winner != evaluation.selected_model.name or winner != evaluation.metadata.get("selected_model"):
        raise ValueError("Evaluation selection is inconsistent. Run a new evaluation before forecasting.")


def forecast_selected_model(
    history: pd.Series, evaluation: EvaluationResult, horizon: int = 30, *,
    source: str, provenance: str, store: int | None, family: str | None,
) -> FutureForecast:
    """Refit the validation winner on all supplied history and forecast beyond it.

    The previously held-out test observations are now known historical demand.
    They may train this separate future fit, but never change model selection or
    the earlier test metrics. No scores are computed for this refit. The function
    receives no later actuals and never modifies the evaluation object/model.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (7, 14, 30):
        raise ValueError("Forecast horizon must be 7, 14, or 30 days.")
    sales = validate_daily_series(history)
    validate_evaluation_context(history, evaluation, source=source, provenance=provenance, store=store, family=family)
    model = fit_model(evaluation.selected_model.name, sales)
    forecasts = predict_recursive(model, sales, horizon)
    expected_dates = pd.date_range(sales.index[-1], periods=int(horizon) + 1, freq="D")[1:]
    if forecasts.date.tolist() != expected_dates.tolist() or not np.isfinite(forecasts.forecast).all() or forecasts.forecast.lt(0).any():
        raise ValueError("Forecast output failed date or demand validation; no forecast was published.")
    forecasts["model"] = model.name
    forecasts["store_nbr"] = store
    forecasts["family"] = family
    forecasts["source"] = source
    forecasts["provenance"] = provenance
    forecasts["units"] = "sales units"
    forecasts["lead_day"] = np.arange(1, int(horizon) + 1)
    forecasts["horizon_days"] = int(horizon)
    forecasts["history_start"] = sales.index[0]
    forecasts["fit_scope"] = "All selected history; includes former evaluation holdouts"
    forecasts["selection_basis"] = "Validation only; test scores not used for selection"
    forecasts["uncertainty_status"] = "unavailable"
    forecasts["evaluation_fingerprint"] = evaluation.metadata["data_fingerprint"]
    metadata = deepcopy(evaluation.metadata)
    metadata.update({
        "forecast_version": FORECAST_VERSION, "horizon_days": int(horizon),
        "evaluation_trained_through": evaluation.metadata["trained_through"],
        "trained_through": str(model.trained_through.date()), "history_start": str(sales.index[0].date()),
        "history_days": len(sales), "training_rows": model.training_rows,
        "forecast_start": str(expected_dates[0].date()), "forecast_end": str(expected_dates[-1].date()),
        "artifact_scope": "Separate full-selected-history future refit, including former test observations; not the assessed holdout fit",
        "uncertainty_status": "unavailable", "uncertainty_reason": INTERVAL_REASON,
        "interval_coverage": None,
        "evaluation_data_fingerprint": evaluation.metadata["data_fingerprint"],
        "evaluation_metrics_apply_to": "Earlier pre-test fit only; no accuracy claim for this refit",
    })
    return FutureForecast(model, forecasts, sales.iloc[-28:].copy(), metadata)


def select_forecast_horizon(result: FutureForecast, horizon: int) -> FutureForecast:
    """Take a supported forecast prefix without refitting or changing stored data."""
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (7, 14, 30):
        raise ValueError("Forecast horizon must be 7, 14, or 30 days.")
    if horizon > len(result.forecasts):
        raise ValueError("Requested horizon exceeds the generated forecast; generate a longer forecast first.")
    rows = result.forecasts.iloc[:horizon].copy()
    rows["horizon_days"] = int(horizon)
    metadata = deepcopy(result.metadata)
    metadata.update(horizon_days=int(horizon), forecast_end=str(rows.date.iloc[-1].date()))
    return FutureForecast(result.model, rows, result.history_tail.copy(), metadata)
