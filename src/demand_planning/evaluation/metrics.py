"""Forecast errors in units and WAPE percent, with explicit zero-demand behavior."""

import numpy as np
from numpy.typing import ArrayLike


def forecast_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float | None]:
    """Score equal-length finite vectors; clip predictions at zero before scoring.

    All-zero actual demand makes WAPE undefined (None), even for a perfect
    forecast. MAE/RMSE remain defined. Missing targets are never silently skipped.
    """
    truth, forecast = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if truth.ndim != 1 or forecast.ndim != 1 or truth.shape != forecast.shape or not truth.size:
        raise ValueError("Metrics require nonempty one-dimensional arrays of equal length.")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all() or (truth < 0).any():
        raise ValueError("Metrics require finite values and nonnegative actual demand.")
    errors = np.maximum(forecast, 0) - truth
    denominator = float(truth.sum())
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "wape_pct": 100 * float(np.abs(errors).sum()) / denominator if denominator > 0 else None}
