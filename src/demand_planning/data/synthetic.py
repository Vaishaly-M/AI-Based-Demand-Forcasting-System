"""Generate illustrative sales; these are not observations from a retailer."""

from datetime import date
from numbers import Integral

import numpy as np
import pandas as pd


def generate_synthetic_sales(
    days: int = 365, seed: int = 42, start_date: str = "2024-01-01"
) -> pd.DataFrame:
    """Return daily sales for two stores and three families using a local RNG.

    Includes trend, weekly/annual seasonality, promotions, noise, and occasional
    zero sales. Inventory and commercial costs are deliberately excluded.
    """
    if isinstance(days, bool) or not isinstance(days, Integral) or not 1 <= days <= 3650:
        raise ValueError("days must be an integer between 1 and 3650.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    try:
        start = date.fromisoformat(start_date)
        dates = pd.date_range(start, periods=int(days), freq="D")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("start_date must be a supported ISO date (YYYY-MM-DD).") from exc

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    families = {"GROCERY I": 80, "DAIRY": 45, "PRODUCE": 60}
    for day_index, timestamp in enumerate(dates):
        weekly = 1.15 if timestamp.dayofweek >= 5 else 1.0
        annual = 1 + 0.15 * np.sin(2 * np.pi * (timestamp.dayofyear - 1) / 365.25)
        trend = 1 + 0.0003 * day_index
        for store in (1, 2):
            for family, base in families.items():
                onpromotion = int(rng.integers(1, 6)) if rng.random() < 0.12 else 0
                mean = base * (1 + 0.12 * (store - 1)) * weekly * annual * trend
                mean *= 1.2 if onpromotion else 1.0
                sales = max(0, round(float(rng.normal(mean, mean * 0.12))))
                if rng.random() < 0.02:
                    sales = 0
                rows.append({
                    "id": len(rows), "date": timestamp.strftime("%Y-%m-%d"),
                    "store_nbr": store, "family": family, "sales": float(sales),
                    "onpromotion": onpromotion, "is_synthetic": True,
                })
    return pd.DataFrame(rows)
