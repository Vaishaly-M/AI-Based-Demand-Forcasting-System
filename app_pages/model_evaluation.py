"""Explicit experiment execution with validation-only selection and test audit."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.evaluation.artifacts import save_evaluation
from src.demand_planning.evaluation.experiment import EXPERIMENT_VERSION, run_evaluation, series_fingerprint
from src.demand_planning.evaluation.splits import chronological_split

st.title("Model Evaluation")
dataset = st.session_state.get("sales_dataset")
selection = st.session_state.get("sales_selection")
if dataset is None or selection is None:
    st.info("Choose a valid dataset, store, family, and date range to evaluate.")
    st.stop()
store, family, start, end = selection
st.subheader(f"Store {store} · {family}")
st.caption(f"Source: {dataset.source}")
st.warning(dataset.provenance)
history = daily_sales(dataset.frame, store, family, start, end)
try:
    split = chronological_split(history)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()
st.dataframe(split.summary().rename(columns={"partition": "Partition", "start": "Start", "end": "End", "days": "Days"}),
             hide_index=True, width="stretch")
st.write("Compare Seasonal Naive, Linear Regression, Random Forest, and XGBoost with fixed settings. Select using 30-day validation WAPE, then refit only the winner through validation end and evaluate it on the reserved test period.")
st.caption("All models use the same forecast origins and 7/14/30-day horizons. WAPE is N/A for all-zero demand; selection then uses MAE. Ties use MAE, RMSE, then the model order above. Predictions feed future lag features; held-out actuals do not.")
st.info("Freeze dates and settings before running. Test results are for final assessment, not choosing another model or repeatedly changing settings. This interactive tool cannot prevent manual reuse of a previously viewed test period.")
signature = (EXPERIMENT_VERSION, series_fingerprint(history), dataset.source, dataset.provenance, store, family)
if st.session_state.get("evaluation_signature") != signature:
    st.session_state["evaluation_result"] = None
    st.session_state["evaluation_signature"] = signature
    st.session_state["evaluation_saved"] = None
result = st.session_state.get("evaluation_result")
if result is None:
    if st.button("Run model evaluation", type="primary"):
        try:
            with st.spinner("Training models and evaluating chronological forecasts..."):
                result = run_evaluation(history, dataset.source, dataset.provenance, store, family)
                st.session_state["evaluation_result"] = result
        except (ValueError, RuntimeError) as exc:
            st.error(f"Evaluation did not complete: {exc}")
            st.stop()
    else:
        st.caption("No experiment has run for this selection. Existing sales charts do not represent model accuracy.")
        st.stop()
st.success(f"Selected on validation: {result.selected_model.name}")
st.subheader("Validation comparison")
st.dataframe(result.validation_metrics[["model", "horizon", "mae", "rmse", "wape_pct"]], hide_index=True, width="stretch")
st.caption("MAE and RMSE are sales units; WAPE is percent. The 7/14-day results are prefixes of the same 30-day forecast, not independent trials. No tuning was performed.")
st.subheader("Earlier rolling-origin checks")
if result.cv_metrics.empty:
    st.info("Training history is too short for an additional 30-day fold after 56 training days. Dedicated validation/test evaluation is still available.")
else:
    st.write("Up to three expanding-window folds entirely inside training. These are diagnostics; the dedicated validation window determines selection.")
    st.dataframe(result.cv_metrics, hide_index=True, width="stretch")
st.subheader("Selected model: reserved test assessment")
st.dataframe(result.test_metrics[["model", "horizon", "mae", "rmse", "wape_pct"]], hide_index=True, width="stretch")
st.caption("Only the selected model was tested. This saved fit excludes test observations. Demand Forecasting can create a separate full-history refit without changing these scores. No calibrated uncertainty intervals are available.")
st.page_link("app_pages/demand_forecasting.py", label="Forecast with the validation-selected model")
stage = st.radio("Actual versus predicted", ["validation", "test"], horizontal=True)
rows = result.predictions.loc[result.predictions.stage.eq(stage)]
actual = rows.drop_duplicates("date")[["date", "actual"]].rename(columns={"actual": "units"})
actual["series"] = "Actual"
predicted = rows[["date", "forecast", "model"]].rename(columns={"forecast": "units", "model": "series"})
chart = pd.concat([actual, predicted], ignore_index=True)
st.plotly_chart(px.line(chart, x="date", y="units", color="series", labels={"units": "Sales units", "date": "Date"}), width="stretch")
with st.expander("Experiment metadata and predictions"):
    st.json(result.metadata)
    st.dataframe(rows, hide_index=True, width="stretch")
if st.session_state.get("evaluation_saved"):
    st.caption(f"Saved local experiment: {st.session_state['evaluation_saved']}")
elif st.button("Save local experiment artifacts"):
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = Path(__file__).resolve().parents[1] / "reports/evaluation" / run_name
    try:
        save_evaluation(result, output)
        st.session_state["evaluation_saved"] = str(output.relative_to(Path(__file__).resolve().parents[1]))
        st.success(f"Saved locally: reports/evaluation/{run_name}")
    except OSError:
        st.error("Could not save this experiment. Check local write permissions and available disk space.")
