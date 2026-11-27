"""Coverage-aware sales dashboard for one store and family."""
import plotly.express as px
import streamlit as st

from src.demand_planning.analytics.sales import daily_sales, family_rankings, period_sales, sales_kpis, seasonal_profiles

st.title("Sales Analytics")
dataset = st.session_state.get("sales_dataset")
selection = st.session_state.get("sales_selection")
if dataset is None:
    st.info("No valid sales dataset loaded. Choose the synthetic demo or provide a valid historical CSV in the sidebar.")
    st.stop()
st.caption(f"Source: {dataset.source}")
st.warning(dataset.provenance)
with st.expander("Data quality and preparation report"):
    st.dataframe(dataset.report, hide_index=True, width="stretch")
    st.write("Coverage below measures missing dates between each series' first and last observation. It does not assume when a store opened or a family became available.")
    st.dataframe(dataset.coverage, hide_index=True, width="stretch")
    st.caption("No rows dropped, duplicates aggregated, or missing sales imputed. Original source files are unchanged.")
if selection is None:
    st.info("Choose both analysis dates in the sidebar.")
    st.stop()
store, family, start, end = selection
daily = daily_sales(dataset.frame, store, family, start, end)
kpis = sales_kpis(daily)
st.subheader(f"Store {store} · {family}")
st.caption(f"{start:%d %b %Y} to {end:%d %b %Y}. Sales are observed units, not revenue or a measure of unconstrained demand.")
if not kpis["observed_days"]:
    st.info("No observed sales in this date range. Widen the dates or choose another family.")
    st.stop()
if kpis["missing_days"]:
    st.warning(f"{kpis['missing_days']} days have no observation. Totals and averages use observed days only; missing days are not zero demand.")
if len(daily) < 14:
    st.info("Fewer than 14 calendar days selected: not enough history for the two-week growth comparison.")
columns = st.columns(4)
columns[0].metric("Observed sales units", f"{kpis['total_sales']:,.1f}")
columns[1].metric("Average daily demand*", f"{kpis['average_daily_demand']:,.2f}")
columns[2].metric("7-day sales growth", "N/A" if kpis["growth_pct"] is None else f"{kpis['growth_pct']:+.1f}%")
columns[3].metric("Observed days", f"{kpis['observed_days']} / {len(daily)}")
st.caption("*Mean observed daily sales, a demand proxy. Growth compares the final 7 calendar days with the preceding 7 within the selected range. N/A means insufficient history, missing days in either window, or a zero previous total.")

st.subheader("Sales trends")
frequency = st.radio("Time interval", ["Daily", "Weekly", "Monthly"], horizontal=True)
periods = period_sales(daily, {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M"}[frequency])
periods["Coverage"] = periods.complete.map({True: "Complete", False: "Incomplete"})
if frequency == "Daily":
    figure = px.line(periods, x="period_start", y="sales", labels={"period_start": "Date", "sales": "Observed sales units"}, color_discrete_sequence=["#167D8D"])
    figure.update_traces(connectgaps=False)
else:
    figure = px.bar(periods, x="period_start", y="sales", color="Coverage", barmode="group",
                    color_discrete_map={"Complete": "#167D8D", "Incomplete": "#B7791F"},
                    hover_data=["observed_days", "expected_days"],
                    labels={"period_start": "Period beginning", "sales": "Observed sales units"})
st.plotly_chart(figure, width="stretch")
st.caption("Weeks run Monday–Sunday. Boundary weeks/months and periods with missing days are incomplete; their observed totals are not comparable with full periods. All-missing totals remain blank.")
with st.expander("View trend table"):
    st.dataframe(periods, hide_index=True, width="stretch")

st.subheader("Seasonal patterns (descriptive)")
weekday, monthly = seasonal_profiles(daily)
left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(weekday, x="weekday", y="average_sales", hover_data=["observed_days"],
                         labels={"weekday": "Weekday", "average_sales": "Mean daily sales"},
                         color_discrete_sequence=["#167D8D"]), width="stretch")
with right:
    st.plotly_chart(px.bar(monthly, x="month", y="average_sales", hover_data=["observed_days"],
                         labels={"month": "Month", "average_sales": "Mean daily sales"},
                         color_discrete_sequence=["#167D8D"]), width="stretch")
st.caption("Means use observed days only; hover for sample sizes. Trend, promotions, missing observations, and unequal samples can explain apparent patterns. These charts do not prove recurring seasonality.")
if daily.notna().sum() < 730:
    st.info("Fewer than two years of observed days: annual seasonality cannot be reliably established from this selection.")

st.subheader("High- and low-demand families")
st.caption("All families in the selected store and date range, independent of the family filter above. Rankings use total sales only for families with complete coverage of the same window. Family units may not be economically comparable.")
rankings = family_rankings(dataset.frame, store, start, end)
eligible = rankings.loc[rankings["rank"].notna()]
if eligible.empty:
    st.info("No family has complete coverage in this window, so comparative rankings are unavailable.")
else:
    high, low = st.columns(2)
    with high:
        st.write("Highest observed demand")
        st.dataframe(eligible.head(5)[["family", "total_sales", "average_daily_sales"]], hide_index=True, width="stretch")
    with low:
        st.write("Lowest observed demand")
        st.dataframe(eligible.sort_values("total_sales").head(5)[["family", "total_sales", "average_daily_sales"]], hide_index=True, width="stretch")
    st.caption("With fewer than ten eligible families, the high and low lists can overlap. Equal totals are ties.")
with st.expander("All families and coverage"):
    st.dataframe(rankings, hide_index=True, width="stretch")
