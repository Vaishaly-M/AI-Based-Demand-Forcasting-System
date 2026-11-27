"""Future refits of validation-selected models and an optional baseline preview."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import plotly.graph_objects as go
import streamlit as st

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.evaluation.experiment import series_fingerprint
from src.demand_planning.forecasting.artifacts import forecast_csv_bytes, save_forecast
from src.demand_planning.forecasting.future import (
    FORECAST_VERSION, INTERVAL_REASON, forecast_selected_model, select_forecast_horizon,
    validate_evaluation_context,
)
from src.demand_planning.forecasting.seasonal_naive import seasonal_naive_forecast

st.title("Demand Forecasting")
st.info("Generate future demand from the validation-selected model, refitted on all selected history. Evaluation scores remain attached to the earlier holdout fit.")
dataset = st.session_state.get("sales_dataset")
selection = st.session_state.get("sales_selection")
if dataset is None or selection is None:
    st.session_state["future_result"] = None
    st.info("Choose a valid sales dataset, store, family, and both history dates in the sidebar.")
    st.stop()
store, family, start, end = selection
st.subheader(f"Store {store} · {family}")
st.caption(f"Source: {dataset.source}")
st.warning(dataset.provenance)
st.write(f"History: {start:%d %b %Y} to {end:%d %b %Y}. Forecast cutoff: {end:%d %b %Y}. Sales after this cutoff are excluded, even if available in the source.")
st.caption("Future means after the data cutoff, not after today's date. Use the latest available end date for the newest forecast, or an earlier date for a historical-origin preview. This page does not score future accuracy.")
method = st.radio("Forecast method", ["Selected model", "Seasonal Naive preview"], horizontal=True, key="forecast_method")
horizon = st.selectbox("Forecast horizon (days)", [7, 14, 30], key="forecast_horizon")
history = daily_sales(dataset.frame, store, family, start, end)
future = None
if method == "Selected model":
    evaluation = st.session_state.get("evaluation_result")
    try:
        if evaluation is None:
            raise ValueError("Run Model Evaluation for the current data and filters first, then return here.")
        validate_evaluation_context(history, evaluation, source=dataset.source, provenance=dataset.provenance,
                                    store=store, family=family)
    except ValueError as exc:
        st.session_state["future_result"] = None
        st.info(str(exc))
        st.page_link("app_pages/model_evaluation.py", label="Open Model Evaluation")
        st.stop()
    signature = (FORECAST_VERSION, series_fingerprint(history), dataset.source, dataset.provenance,
                 store, family, evaluation.selected_model.name)
    if st.session_state.get("future_signature") != signature:
        st.session_state["future_signature"] = signature
        st.session_state["future_result"] = None
        st.session_state["future_saved"] = None
    if st.session_state.get("future_result") is None:
        st.write(f"Validation selected {evaluation.selected_model.name}. The new fit will use all {len(history)} selected history days, including the former evaluation holdouts.")
        if st.button("Refit selected model and forecast", type="primary"):
            try:
                with st.spinner("Refitting the selected model and forecasting future demand..."):
                    st.session_state["future_result"] = forecast_selected_model(
                        history, evaluation, 30, source=dataset.source, provenance=dataset.provenance,
                        store=store, family=family,
                    )
            except (ValueError, RuntimeError) as exc:
                st.error(f"Future forecast could not complete: {exc}")
                st.stop()
        else:
            st.stop()
    future = select_forecast_horizon(st.session_state["future_result"], horizon)
    forecast = future.forecasts
    st.success(f"Future model: {future.model.name}; refit through {end:%d %b %Y}.")
    st.caption(f"Earlier assessed fit ended {future.metadata['evaluation_trained_through']}. Its test scores do not measure this new full-history fit or the unobserved future. Changing the horizon uses the same forecast prefix without retraining.")
else:
    try:
        forecast = seasonal_naive_forecast(history, horizon)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    forecast["store_nbr"] = store
    forecast["family"] = family
    forecast["source"] = dataset.source
    forecast["provenance"] = dataset.provenance
    forecast["units"] = "sales units"
    forecast["horizon_days"] = horizon
    forecast["fit_scope"] = "Unselected weekly baseline preview"
    forecast["uncertainty_status"] = "unavailable"
    st.write("Unselected baseline preview: repeat the final observed week's sales on matching weekdays. This does not learn trends, promotions, or annual seasonality.")
left, right = st.columns(2)
left.metric("Forecast units over horizon", f"{forecast.forecast.sum():,.1f}")
right.metric("Mean forecast units per day", f"{forecast.forecast.mean():,.2f}")
figure = go.Figure()
shown = history.iloc[-90:]
figure.add_trace(go.Scatter(x=shown.index, y=shown, mode="lines", name="Observed sales", connectgaps=False,
                           line={"color": "#167D8D"}))
figure.add_trace(go.Scatter(x=forecast.date, y=forecast.forecast, mode="lines", name=str(forecast.model.iloc[0]),
                           line={"color": "#B7791F", "dash": "dash"}))
figure.update_layout(xaxis_title="Date", yaxis_title="Sales units", legend_title_text="Series")
st.plotly_chart(figure, width="stretch")
st.caption("Chart shows up to 90 history days. Point forecasts are estimates of observed sales, not guaranteed demand or purchasing recommendations.")
st.info(INTERVAL_REASON)
st.dataframe(forecast[["date", "forecast", "forecast_origin", "model"]].rename(columns={"date": "Date", "forecast": "Forecast units",
                                    "forecast_origin": "Forecast cutoff", "model": "Method"}),
             hide_index=True, width="stretch")
st.download_button("Download forecast CSV", forecast_csv_bytes(forecast),
                   file_name=f"demand_forecast_store_{store}_{end:%Y%m%d}_{horizon}d.csv", mime="text/csv")
st.caption("CSV includes source/provenance, store/family, method, cutoff, units, horizon, and uncertainty status.")
if future is not None:
    with st.expander("Future refit details"):
        st.json(future.metadata)
    saved = st.session_state.get("future_saved")
    if saved and saved[0] == horizon:
        st.caption(f"Saved local forecast: {saved[1]}")
    elif st.button("Save future model and forecast locally"):
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        output = Path(__file__).resolve().parents[1] / "reports/forecasts" / run_name
        try:
            save_forecast(future, output)
            relative = f"reports/forecasts/{run_name}"
            st.session_state["future_saved"] = (horizon, relative)
            st.success(f"Saved future artifacts: {relative}")
        except (OSError, ValueError):
            st.error("Could not save the forecast. Check local permissions and available disk space; existing folders are never overwritten.")
