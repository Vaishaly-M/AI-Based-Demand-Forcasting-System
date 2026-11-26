"""Streamlit entry point and shared navigation for the Phase 1 scaffold."""

import streamlit as st

st.set_page_config(page_title="Demand & Inventory Planning", page_icon="📊", layout="wide")
st.sidebar.title("Demand & Inventory")
st.sidebar.caption("Portfolio project · Phase 1")
st.sidebar.selectbox("Store", ["Available in Phase 2"], disabled=True)
st.sidebar.selectbox("Product family", ["Available in Phase 2"], disabled=True)
st.sidebar.info("No business dataset loaded. Demo sales are synthetic.")

navigation = st.navigation([
    st.Page("app_pages/overview.py", title="Overview", default=True),
    st.Page("app_pages/sales_analytics.py", title="Sales Analytics"),
    st.Page("app_pages/demand_forecasting.py", title="Demand Forecasting"),
    st.Page("app_pages/model_evaluation.py", title="Model Evaluation"),
    st.Page("app_pages/inventory_planning.py", title="Inventory Planning"),
    st.Page("app_pages/business_insights.py", title="Business Insights"),
])
navigation.run()
