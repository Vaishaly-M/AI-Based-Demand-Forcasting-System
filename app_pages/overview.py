"""Project landing page; no calculated business metrics in Phase 1."""

import streamlit as st

st.title("AI-Based Demand Forecasting & Inventory Planning")
st.write("Explore retail demand and support replenishment decisions with transparent assumptions.")
st.info("Phase 1: project foundation. Analytics, forecasts, and inventory calculations are planned features.")
left, middle, right = st.columns(3)
left.metric("Development phase", "1 of 8")
middle.metric("Business data", "Not loaded")
right.metric("Trained models", "None")
st.subheader("Start with reproducible demo sales")
st.code("python -m scripts.generate_synthetic_data", language="powershell")
st.write("This creates data/synthetic/sales.csv locally. Loading and analysing that file will be added in Phase 2.")
st.caption("Sales represent units, not revenue. Inventory levels, supplier lead times, and costs must be supplied separately.")
