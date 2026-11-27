"""Project status and data provenance overview."""

import streamlit as st

st.title("AI-Based Demand Forecasting & Inventory Planning")
st.write("Explore retail demand and support replenishment decisions with transparent assumptions.")
st.info("Phase 6: sales analytics, evaluated forecasts, and inventory scenarios with user-supplied stock, supplier, and cost assumptions. Business Insights integration remains planned.")
dataset = st.session_state.get("sales_dataset")
left, middle, right = st.columns(3)
left.metric("Development phase", "6 of 8")
middle.metric("Validated sales rows", f"{len(dataset.frame):,}" if dataset is not None else "Not loaded")
right.metric("Evaluation", "Run on Model Evaluation page")
if dataset is not None:
    st.write(f"Source: {dataset.source}")
    st.warning(dataset.provenance)
st.subheader("Explore your sales")
st.write("Choose a data source, store, product family, and date range in the sidebar, then open Sales Analytics. The built-in demonstration works without a download.")
st.page_link("app_pages/sales_analytics.py", label="Open Sales Analytics")
st.page_link("app_pages/demand_forecasting.py", label="Generate future demand forecasts")
st.page_link("app_pages/model_evaluation.py", label="Compare forecasting models")
st.page_link("app_pages/inventory_planning.py", label="Plan an inventory scenario")
st.write("To use the real dataset, place Kaggle train.csv in data/raw/ and choose the local source, or upload a historical sales CSV. Invalid rows must be corrected before analysis.")
st.caption("Sales represent units, not revenue. Inventory levels, supplier lead times, and costs must be supplied separately.")
