"""Hand-calculated analytics checks for gaps, calendar periods, and comparisons."""

import numpy as np
import pandas as pd
import pytest

from src.demand_planning.analytics.sales import daily_sales, family_rankings, period_sales, sales_kpis, seasonal_profiles


def test_daily_calendar_keeps_missing_distinct_from_zero() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
                          "store_nbr": [1, 1], "family": ["A", "A"], "sales": [0., 4.]})
    daily = daily_sales(frame, 1, "A", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05"))
    assert len(daily) == 5
    assert daily.isna().tolist() == [True, False, True, False, True]
    assert sales_kpis(daily) == {"total_sales": 4., "average_daily_demand": 2.,
                                "observed_days": 2, "missing_days": 3, "growth_pct": None}
    assert sales_kpis(daily_sales(frame, 2, "A", daily.index.min(), daily.index.max()))["total_sales"] is None
    with pytest.raises(ValueError):
        daily_sales(frame, 1, "A", daily.index.max(), daily.index.min())


def test_growth_equal_windows_and_zero_denominator() -> None:
    daily = pd.Series([2.] * 7 + [3.] * 7, index=pd.date_range("2024-01-01", periods=14))
    assert sales_kpis(daily)["growth_pct"] == 50.
    assert sales_kpis(daily.iloc[1:])["growth_pct"] is None
    daily.iloc[0] = np.nan
    assert sales_kpis(daily)["growth_pct"] is None
    daily.iloc[:7] = 0
    assert sales_kpis(daily)["growth_pct"] is None
    daily.iloc[:7] = 2
    daily.iloc[-7:] = 0
    assert sales_kpis(daily)["growth_pct"] == -100.


def test_periods_label_boundaries_and_missing_days() -> None:
    daily = pd.Series(1., index=pd.date_range("2024-01-02", "2024-01-15"))
    weekly = period_sales(daily, "W-SUN")
    assert weekly.sales.tolist() == [6., 7., 1.]
    assert weekly.complete.tolist() == [False, True, False]
    assert weekly.expected_days.tolist() == [7, 7, 7]
    daily.iloc[6:13] = np.nan
    weekly = period_sales(daily, "W-SUN")
    assert pd.isna(weekly.sales.iloc[1])
    assert not weekly.complete.any()


def test_leap_month_and_empty_periods() -> None:
    daily = pd.Series(2., index=pd.date_range("2024-02-01", "2024-02-29"))
    month = period_sales(daily, "M").iloc[0]
    assert month.sales == 58.
    assert month.expected_days == 29
    assert month.complete
    assert not period_sales(daily.iloc[1:], "M").complete.any()
    assert period_sales(daily.iloc[:0], "M").empty
    with pytest.raises(ValueError):
        period_sales(daily, "invalid")


def test_seasonal_means_include_zeros_but_exclude_missing() -> None:
    daily = pd.Series([0., np.nan, 4., 4., 4., 4., 4., 8.], index=pd.date_range("2024-01-01", periods=8))
    weekday, monthly = seasonal_profiles(daily)
    assert weekday.iloc[0].average_sales == 4.
    assert weekday.iloc[0].observed_days == 2
    assert weekday.iloc[1].observed_days == 0
    assert pd.isna(weekday.iloc[1].average_sales)
    assert monthly.iloc[0].observed_days == 7
    assert monthly.iloc[1].observed_days == 0


def test_rankings_exclude_incomplete_families_and_other_stores() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2 + ["2024-01-01", "2024-01-01"]),
        "store_nbr": [1, 1, 1, 1, 1, 2], "family": ["A", "A", "B", "B", "C", "A"],
        "sales": [2., 2., 0., 0., 100., 1000.],
    })
    ranks = family_rankings(frame, 1, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")).set_index("family")
    assert ranks.loc["A", "rank"] == 1
    assert ranks.loc["B", "rank"] == 2
    assert pd.isna(ranks.loc["C", "rank"])
    assert ranks.loc["C", "coverage_pct"] == 50.
    assert ranks.loc["A", "total_sales"] == 4.
