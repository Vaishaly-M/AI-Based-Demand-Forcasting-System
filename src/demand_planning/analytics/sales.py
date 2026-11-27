"""Coverage-aware calculations for a selected daily store/family series."""

import pandas as pd


def daily_sales(
    frame: pd.DataFrame, store: int, family: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series:
    """Return a full calendar with NaN for unobserved days, including edges."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Choose a valid start date on or before the end date.")
    selected = frame.loc[frame.store_nbr.eq(store) & frame.family.eq(family)]
    values = selected.set_index("date").sales
    return values.reindex(pd.date_range(start, end, name="date")).rename("sales")


def sales_kpis(daily: pd.Series) -> dict[str, float | int | None]:
    """Observed-only totals/means; growth needs two complete seven-day windows."""
    observed = int(daily.notna().sum())
    growth = None
    if len(daily) >= 14 and daily.iloc[-14:].notna().all():
        previous = float(daily.iloc[-14:-7].sum())
        if previous > 0:
            growth = 100 * (float(daily.iloc[-7:].sum()) - previous) / previous
    return {"total_sales": float(daily.sum()) if observed else None,
            "average_daily_demand": float(daily.mean()) if observed else None,
            "observed_days": observed, "missing_days": len(daily) - observed,
            "growth_pct": growth}


def period_sales(daily: pd.Series, frequency: str) -> pd.DataFrame:
    """Aggregate observed sales; mark boundary periods and gaps as incomplete.

    Weekly bins run Monday–Sunday. Partial totals remain explicit and must not
    be interpreted as full-period comparisons; all-missing totals stay NaN.
    """
    if frequency not in {"D", "W-SUN", "M"}:
        raise ValueError("frequency must be D, W-SUN, or M.")
    if daily.empty:
        return pd.DataFrame(columns=["period_start", "sales", "observed_days", "expected_days", "complete"])
    grouped = daily.groupby(daily.index.to_period(frequency))
    result = grouped.agg(sales=lambda values: values.sum(min_count=1), observed_days="count")
    result["expected_days"] = [int((period.end_time.normalize() - period.start_time).days) + 1 for period in result.index]
    result["complete"] = result.observed_days.eq(result.expected_days)
    result.insert(0, "period_start", result.index.start_time)
    return result.reset_index(drop=True)


def seasonal_profiles(daily: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return descriptive weekday/month means and sample sizes, not forecasts."""
    weekday = daily.groupby(daily.index.dayofweek).agg(["mean", "count"]).reindex(range(7))
    weekday.insert(0, "weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    monthly = daily.groupby(daily.index.month).agg(["mean", "count"]).reindex(range(1, 13))
    monthly.insert(0, "month", pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%b"))
    for profile in (weekday, monthly):
        profile["count"] = profile["count"].fillna(0).astype(int)
        profile.rename(columns={"mean": "average_sales", "count": "observed_days"}, inplace=True)
    return weekday.reset_index(drop=True), monthly.reset_index(drop=True)


def family_rankings(frame: pd.DataFrame, store: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Compare families over one common window; rank only complete histories."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("Start date must not be after end date.")
    store_rows = frame.loc[frame.store_nbr.eq(store)]
    families = store_rows.family.drop_duplicates()
    window = store_rows.loc[store_rows.date.between(start, end)]
    summary = window.groupby("family", observed=True).agg(
        total_sales=("sales", "sum"), observed_days=("sales", "count"),
        average_daily_sales=("sales", "mean"),
    ).reindex(families)
    summary["observed_days"] = summary.observed_days.fillna(0).astype(int)
    summary["expected_days"] = (end - start).days + 1
    summary["coverage_pct"] = 100 * summary.observed_days / summary.expected_days
    summary["rank"] = summary.total_sales.where(summary.observed_days.eq(summary.expected_days)).rank(
        ascending=False, method="min"
    ).astype("Int64")
    return summary.sort_values(["rank", "total_sales"], ascending=[True, False], na_position="last").reset_index()
