"""Forecast CSV and trusted local future-model bundles, separate from evaluation."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.demand_planning.forecasting.future import FutureForecast


def forecast_csv_bytes(forecasts: pd.DataFrame) -> bytes:
    """Encode forecast rows with provenance and ISO dates for CSV download."""
    return forecasts.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")


def save_forecast(result: FutureForecast, directory: Path) -> Path:
    """Create a new directory with future CSV, metadata, and trusted Joblib fit.

    Existing runs are never overwritten. This bundle has different names/scope
    from evaluation artifacts. Do not load Joblib files from untrusted sources.
    """
    metadata = json.dumps(result.metadata, indent=2, allow_nan=False)
    csv_data = forecast_csv_bytes(result.forecasts)
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": result.model, "metadata": result.metadata, "history_tail": result.history_tail},
                directory / "future_model.joblib")
    (directory / "forecast_metadata.json").write_text(metadata, encoding="utf-8")
    (directory / "forecast.csv").write_bytes(csv_data)
    return directory
