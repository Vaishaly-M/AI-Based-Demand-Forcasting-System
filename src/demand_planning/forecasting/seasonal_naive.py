"""Weekly seasonal naive benchmark with fixed-origin multi-step forecasts."""

from numbers import Integral

import numpy as np
import pandas as pd

from src.demand_planning.data.time_series import validate_daily_series

SUPPORTED_HORIZONS = (7, 14, 30)


def seasonal_naive_forecast(history: pd.Series, horizon: int = 7) -> pd.DataFrame:
    """Repeat the final observed week for a 7-, 14-, or 30-day horizon.

    The origin is the last history date. For h > 7, repeating the last week
    equals recursion using the prediction from seven days earlier. Actuals
    after the origin are neither required nor fetched. No intervals are inferred.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon not in SUPPORTED_HORIZONS:
        raise ValueError("Forecast horizon must be 7, 14, or 30 days.")
    sales = validate_daily_series(history)
    if len(sales) < 7 or sales.iloc[-7:].isna().any():
        raise ValueError("Seasonal Naive requires seven consecutive observed days ending at the forecast cutoff. Widen the history or choose a cutoff with a complete final week.")
    origin = sales.index[-1]
    values = np.resize(sales.iloc[-7:].to_numpy(), int(horizon))
    return pd.DataFrame({
        "date": pd.date_range(origin, periods=int(horizon) + 1, freq="D")[1:],
        "forecast": np.maximum(values, 0),
        "forecast_origin": origin,
        "model": "Seasonal Naive (7-day)",
    })
