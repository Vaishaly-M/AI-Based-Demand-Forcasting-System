"""One-step historical features; never a precomputed multi-step holdout matrix."""

import numpy as np
import pandas as pd

from src.demand_planning.data.time_series import validate_daily_series

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)
WARMUP_DAYS = 28


def build_features(history: pd.Series) -> pd.DataFrame:
    """Build features at date t using only sales strictly before t.

    Preserve warm-up/gap NaNs; no fitting, filling, or dropping is performed.
    This matrix is appropriate for historical one-step training examples.
    Multi-step validation must instead rebuild each next-day row from the
    forecast-origin history plus prior predictions, never held-out actuals.
    """
    sales = validate_daily_series(history)
    dates = sales.index
    features = pd.DataFrame(index=dates)
    features["day_of_week"] = dates.dayofweek
    features["day_of_month"] = dates.day
    features["month"] = dates.month
    features["day_of_year"] = dates.dayofyear
    features["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    features["weekday_sin"] = np.sin(2 * np.pi * dates.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * dates.dayofweek / 7)
    for lag in LAGS:
        features[f"lag_{lag}"] = sales.shift(lag)
    past = sales.shift(1)
    for window in ROLLING_WINDOWS:
        rolling = past.rolling(window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean()
        features[f"rolling_std_{window}"] = rolling.std(ddof=0)
    return features


def next_day_features(history: pd.Series) -> pd.DataFrame:
    """Build one future row from known history, optionally including predictions.

    Require all 28 most recent days to be observed/predicted. The caller must
    append each predicted value before requesting another multi-step row.
    No future targets are accepted as a separate input.
    """
    sales = validate_daily_series(history)
    if len(sales) < WARMUP_DAYS or sales.iloc[-WARMUP_DAYS:].isna().any():
        raise ValueError("Next-day features require 28 consecutive known daily values; missing sales are not imputed.")
    next_date = pd.date_range(sales.index[-1], periods=2, freq="D")[1]
    extended = pd.concat([sales, pd.Series([np.nan], index=pd.DatetimeIndex([next_date]))])
    return build_features(extended).iloc[[-1]]
