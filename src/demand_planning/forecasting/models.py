"""Fixed model definitions and training-only preprocessing for reproducible runs."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.demand_planning.data.time_series import validate_daily_series
from src.demand_planning.features.calendar_lags import build_features, next_day_features
from src.demand_planning.forecasting.seasonal_naive import seasonal_naive_forecast

MODEL_NAMES = ("Seasonal Naive", "Linear Regression", "Random Forest", "XGBoost")
RANDOM_SEED = 42


@dataclass
class FittedModel:
    """Trusted local model and its training/feature contract."""

    name: str
    estimator: Any
    feature_columns: tuple[str, ...]
    trained_through: pd.Timestamp
    training_rows: int


def make_estimator(name: str) -> Any:
    """Create a fresh estimator; no hyperparameters depend on validation/test."""
    if name == "Linear Regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2,
                                     random_state=RANDOM_SEED, n_jobs=1)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05,
                            subsample=1.0, colsample_bytree=1.0, objective="reg:squarederror",
                            tree_method="hist", random_state=RANDOM_SEED, n_jobs=1)
    raise ValueError(f"Unknown machine learning model: {name}.")


def fit_model(name: str, history: pd.Series) -> FittedModel:
    """Fit only on the supplied history; remove exactly the feature warm-up rows."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}.")
    sales = validate_daily_series(history)
    if len(sales) < 56 or sales.isna().any():
        raise ValueError("Model comparisons require at least 56 complete training days.")
    if name == "Seasonal Naive":
        return FittedModel(name, None, (), sales.index[-1], len(sales))
    features = build_features(sales)
    ready = features.notna().all(axis=1)
    estimator = make_estimator(name)
    estimator.fit(features.loc[ready], sales.loc[ready])
    return FittedModel(name, estimator, tuple(features.columns), sales.index[-1], int(ready.sum()))


def predict_recursive(model: FittedModel, history: pd.Series, horizon: int = 30) -> pd.DataFrame:
    """Forecast from history only; feed clipped predictions back into future lags.

    Last 28 days suffice for the feature specification. No actual targets beyond
    the history endpoint can enter this function. Rebuild features each step.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in (7, 14, 30):
        raise ValueError("Forecast horizon must be 7, 14, or 30 days.")
    sales = validate_daily_series(history)
    if model.trained_through > sales.index[-1]:
        raise ValueError("Model was trained after this forecast origin; this would leak future observations.")
    if model.name == "Seasonal Naive":
        return seasonal_naive_forecast(sales, horizon)
    values = sales.iloc[-28:].copy()
    dates: list[pd.Timestamp] = []
    predictions: list[float] = []
    for _ in range(int(horizon)):
        row = next_day_features(values)
        if tuple(row.columns) != model.feature_columns:
            raise ValueError("Model feature contract does not match the current feature builder.")
        prediction = float(model.estimator.predict(row)[0])
        if not np.isfinite(prediction):
            raise ValueError(f"{model.name} produced a nonfinite prediction; evaluation aborted.")
        prediction = max(0., prediction)
        dates.append(row.index[0])
        predictions.append(prediction)
        values = pd.concat([values, pd.Series([prediction], index=row.index)]).iloc[-28:]
    return pd.DataFrame({"date": dates, "forecast": predictions,
                         "forecast_origin": sales.index[-1], "model": model.name})
