"""Chronological raw-series boundaries shared by baseline and ML evaluation."""

from dataclasses import dataclass
from numbers import Integral

import pandas as pd

from src.demand_planning.data.time_series import validate_daily_series


@dataclass(frozen=True)
class TemporalSplit:
    """Non-overlapping chronological observations, not precomputed features."""

    train: pd.Series
    validation: pd.Series
    test: pd.Series

    def summary(self) -> pd.DataFrame:
        """Return explicit inclusive boundaries without exposing holdout targets."""
        return pd.DataFrame([
            {"partition": name, "start": values.index[0], "end": values.index[-1], "days": len(values)}
            for name, values in (("train", self.train), ("validation", self.validation), ("test", self.test))
        ])


def chronological_split(
    history: pd.Series, validation_days: int = 30, test_days: int = 30,
    min_train_days: int = 56,
) -> TemporalSplit:
    """Reserve final test days and preceding validation days, without shuffling.

    Holdouts must each support the largest 30-day horizon. Require at least 56
    training days: 28 warm-up days plus at least 28 usable training rows for the
    default features. This is a mechanical minimum, not evidence of ML adequacy.
    Missing histories are rejected pending an explicit later modelling policy.
    """
    for name, value, minimum in (("validation_days", validation_days, 30),
                                  ("test_days", test_days, 30), ("min_train_days", min_train_days, 56)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    sales = validate_daily_series(history)
    if sales.isna().any():
        raise ValueError("Evaluation splits require complete daily observations; missing sales are not imputed.")
    needed = min_train_days + validation_days + test_days
    if len(sales) < needed:
        raise ValueError(f"At least {needed} daily observations are required for these chronological splits.")
    train_end = len(sales) - validation_days - test_days
    validation_end = len(sales) - test_days
    return TemporalSplit(sales.iloc[:train_end].copy(), sales.iloc[train_end:validation_end].copy(),
                         sales.iloc[validation_end:].copy())
