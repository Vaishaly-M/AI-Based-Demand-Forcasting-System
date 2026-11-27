"""Streamlit entry point, shared sales sources, filters, and navigation."""

import streamlit as st

from app_pages.data_controls import render_data_controls

st.set_page_config(page_title="Demand & Inventory Planning", page_icon="📊", layout="wide")
st.sidebar.title("Demand & Inventory")
st.sidebar.caption("Portfolio project · Phase 6")
render_data_controls()

navigation = st.navigation([
    st.Page("app_pages/overview.py", title="Overview", default=True),
    st.Page("app_pages/sales_analytics.py", title="Sales Analytics"),
    st.Page("app_pages/demand_forecasting.py", title="Demand Forecasting"),
    st.Page("app_pages/model_evaluation.py", title="Model Evaluation"),
    st.Page("app_pages/inventory_planning.py", title="Inventory Planning"),
    st.Page("app_pages/business_insights.py", title="Business Insights"),
])
navigation.run()
