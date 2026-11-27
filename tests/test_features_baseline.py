"""Hand-calculated boundaries and adversarial leakage checks for Phase 3."""

import numpy as np
import pandas as pd
import pytest

from src.demand_planning.data.time_series import validate_daily_series
from src.demand_planning.evaluation.splits import chronological_split
from src.demand_planning.features.calendar_lags import build_features, next_day_features
from src.demand_planning.forecasting.seasonal_naive import seasonal_naive_forecast


def series(days: int = 140) -> pd.Series:
    """Known increasing demand with a daily calendar crossing leap day."""
    return pd.Series(np.arange(1, days + 1, dtype=float), index=pd.date_range("2024-01-01", periods=days))


def test_feature_values_and_warmup() -> None:
    features = build_features(series())
    row = features.iloc[28]
    assert row.lag_1 == 28
    assert row.lag_7 == 22
    assert row.lag_14 == 15
    assert row.lag_28 == 1
    assert row.rolling_mean_7 == 25
    assert row.rolling_mean_28 == 14.5
    assert row.rolling_std_7 == pytest.approx(np.std(np.arange(22, 29)))
    assert features.iloc[:28].isna().any(axis=1).all()
    assert features.iloc[28:].notna().all(axis=1).all()
    assert features.loc["2024-02-29", "day_of_year"] == 60
    assert features.loc["2024-01-06", "is_weekend"] == 1
    assert features.loc["2024-01-01", "weekday_sin"] == 0


def test_current_and_future_sales_cannot_change_features_at_target() -> None:
    original = series()
    altered = original.copy()
    altered.iloc[40:] = 999999
    pd.testing.assert_frame_equal(build_features(original).iloc[:41], build_features(altered).iloc[:41])
    assert build_features(original).iloc[41].lag_1 != build_features(altered).iloc[41].lag_1
    pd.testing.assert_series_equal(original, series())


def test_missing_values_are_not_filled_or_time_compressed() -> None:
    history = series(40)
    history.iloc[30] = np.nan
    features = build_features(history)
    assert pd.isna(features.iloc[31].lag_1)
    assert pd.isna(features.iloc[37].rolling_mean_7)
    assert pd.notna(features.iloc[38].rolling_mean_7)
    assert len(features) == len(history)
    with pytest.raises(ValueError, match="consecutive calendar"):
        build_features(history.dropna())
    with pytest.raises(ValueError, match="28 consecutive"):
        next_day_features(history)


def test_next_day_matches_training_row_without_target_and_uses_predictions() -> None:
    history = series(28)
    row = next_day_features(history)
    assert row.index[0] == pd.Timestamp("2024-01-29")
    pd.testing.assert_frame_equal(row, build_features(series(29)).iloc[[-1]], check_freq=False)
    # A subsequent step uses the previous prediction, not a future actual.
    extended = pd.concat([history, pd.Series([500.], index=row.index)])
    following = next_day_features(extended)
    assert following.iloc[0].lag_1 == 500.
    assert following.iloc[0].rolling_mean_7 == pytest.approx((sum(range(23, 29)) + 500) / 7)
    with pytest.raises(ValueError, match="28 consecutive"):
        next_day_features(series(27))


@pytest.mark.parametrize("horizon", [7, 14, 30])
def test_baseline_repeats_last_week_on_correct_future_dates(horizon: int) -> None:
    history = series(14)
    forecast = seasonal_naive_forecast(history, horizon)
    np.testing.assert_array_equal(forecast.forecast, np.resize(np.arange(8., 15.), horizon))
    assert forecast.date.tolist() == list(pd.date_range("2024-01-15", periods=horizon))
    assert forecast.forecast_origin.eq(history.index[-1]).all()
    assert forecast.forecast.ge(0).all()
    pd.testing.assert_series_equal(history, series(14))


def test_zero_demand_and_earlier_gaps_are_supported_by_baseline() -> None:
    history = series(14)
    history.iloc[:7] = np.nan
    history.iloc[-7:] = 0.
    assert seasonal_naive_forecast(history, 30).forecast.eq(0).all()
    history.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="seven consecutive"):
        seasonal_naive_forecast(history)
    with pytest.raises(ValueError, match="seven consecutive"):
        seasonal_naive_forecast(series(6))


@pytest.mark.parametrize("horizon", [0, -1, 8, 7.0, True, "7", None])
def test_invalid_horizons(horizon: object) -> None:
    with pytest.raises(ValueError, match="horizon"):
        seasonal_naive_forecast(series(), horizon)


@pytest.mark.parametrize("bad", [
    pd.Series(dtype=float), pd.Series([1, 2]), series().iloc[::-1],
    pd.Series([1., 2.], index=pd.to_datetime(["2024-01-01", "2024-01-01"])),
    pd.Series([1.], index=pd.to_datetime(["2024-01-01 12:00"])),
    pd.Series([1.], index=pd.date_range("2024-01-01", periods=1, tz="UTC")),
    pd.Series([-1.], index=pd.date_range("2024-01-01", periods=1)),
    pd.Series([np.inf], index=pd.date_range("2024-01-01", periods=1)),
    pd.Series(["bad"], index=pd.date_range("2024-01-01", periods=1)),
])
def test_invalid_daily_series_rejected(bad: pd.Series) -> None:
    with pytest.raises(ValueError):
        validate_daily_series(bad)


def test_split_boundaries_and_minimum_feature_rows() -> None:
    history = series(116)
    split = chronological_split(history)
    assert [len(split.train), len(split.validation), len(split.test)] == [56, 30, 30]
    assert split.train.index[-1] < split.validation.index[0] < split.test.index[0]
    pd.testing.assert_series_equal(pd.concat([split.train, split.validation, split.test]), history)
    assert len(build_features(split.train).dropna()) == 28
    assert split.summary().days.tolist() == [56, 30, 30]
    split.train.iloc[0] = 12345
    assert history.iloc[0] == 1  # partitions cannot mutate caller history
    with pytest.raises(ValueError, match="116"):
        chronological_split(series(115))


def test_holdout_perturbations_do_not_change_training_or_origin_forecast() -> None:
    original = series()
    split = chronological_split(original)
    altered = original.copy()
    altered.loc[split.validation.index[0]:] = 99999
    changed = chronological_split(altered)
    pd.testing.assert_frame_equal(build_features(split.train), build_features(changed.train))
    pd.testing.assert_frame_equal(seasonal_naive_forecast(split.train, 30), seasonal_naive_forecast(changed.train, 30))
    # Final test predictions may use already observed validation history, never test actuals.
    altered = original.copy()
    altered.loc[split.test.index[0]:] = 99999
    changed = chronological_split(altered)
    pd.testing.assert_frame_equal(
        seasonal_naive_forecast(pd.concat([split.train, split.validation]), 30),
        seasonal_naive_forecast(pd.concat([changed.train, changed.validation]), 30),
    )


@pytest.mark.parametrize("kwargs", [{"validation_days": 29}, {"test_days": 0},
                                    {"min_train_days": 55}, {"test_days": True}, {"validation_days": 30.5}])
def test_split_parameter_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        chronological_split(series(), **kwargs)


def test_split_rejects_unknown_targets() -> None:
    history = series()
    history.iloc[70] = np.nan
    with pytest.raises(ValueError, match="complete daily"):
        chronological_split(history)
