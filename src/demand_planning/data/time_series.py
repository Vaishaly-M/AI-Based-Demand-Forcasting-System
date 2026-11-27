"""Daily series contract shared by features, splits, and baseline forecasts."""

import numpy as np
import pandas as pd


def validate_daily_series(series: pd.Series) -> pd.Series:
    """Return a numeric copy; require a sorted, unique, naive daily calendar.

    Missing values may be explicit NaN, but missing dates must first be reindexed
    by the caller. Positional lags must never mistake two observations for two
    consecutive calendar days. Recorded demand must be finite and nonnegative.
    """
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError("Supply a nonempty daily sales Series.")
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.tz is not None:
        raise ValueError("Sales require a timezone-naive DatetimeIndex without missing dates.")
    if not index.is_unique or not index.is_monotonic_increasing or not index.equals(index.normalize()):
        raise ValueError("Sales dates must be unique, sorted, and at midnight.")
    if not index.equals(pd.date_range(index[0], index[-1], freq="D")):
        raise ValueError("Reindex to consecutive calendar days; preserve unknown sales as NaN.")
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sales must be numeric or explicitly missing.") from exc
    if np.isinf(values).any() or values.lt(0).any():
        raise ValueError("Observed sales must be finite and nonnegative.")
    return values.copy()
