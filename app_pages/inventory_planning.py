"""User-entered inventory scenarios with visible assumptions and timing rules."""
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.evaluation.experiment import series_fingerprint
from src.demand_planning.forecasting.future import FORECAST_VERSION
from src.demand_planning.inventory.planning import InventoryInputs, plan_inventory

st.title("Inventory Planning")
st.write("Explore replenishment using your stock, supplier, and cost assumptions. Sales files do not provide these inventory facts.")
dataset = st.session_state.get("sales_dataset")
selection = st.session_state.get("sales_selection")
if dataset is None or selection is None:
    st.session_state["inventory_plan"] = None
    st.info("Choose a valid sales source, store, family, and both history dates first.")
    st.stop()
store, family, start, end = selection
history = daily_sales(dataset.frame, store, family, start, end)
fingerprint = series_fingerprint(history)
st.subheader(f"Store {store} · {family}")
st.caption(f"Sales source: {dataset.source} | {dataset.provenance}")
st.warning(f"Scenario date: {end:%d %b %Y}, the selected sales cutoff, not necessarily today. Enter inventory and orders as of that date. Product-family units and costs are illustrative; confirm SKU details before purchasing.")
mode = st.radio("Demand basis", ["Forecast-based planning", "Historical mean scenario"], key="inventory_demand_basis", horizontal=True)
future = None
horizon = 30
forecast_signature = None
if mode == "Forecast-based planning":
    future = st.session_state.get("future_result")
    expected = {"source": dataset.source, "provenance": dataset.provenance, "store": store, "family": family,
                "data_fingerprint": fingerprint, "forecast_version": FORECAST_VERSION, "trained_through": str(end.date())}
    if future is None or any(future.metadata.get(key) != value for key, value in expected.items()):
        st.session_state["inventory_plan"] = None
        st.info("Generate a matching selected-model forecast for the current data and filters first, or explicitly choose Historical mean scenario.")
        st.page_link("app_pages/demand_forecasting.py", label="Open Demand Forecasting")
        st.stop()
    horizon = st.selectbox("Forecast demand window (days)", [7, 14, 30], index=2, key="inventory_horizon")
    forecast_signature = (future.model.name, series_fingerprint(future.forecasts.set_index("date").forecast))
    st.caption(f"Demand uses the next {horizon} forecast days from {future.model.name}. This window is independent of the chart horizon on Demand Forecasting.")
else:
    st.info("Historical-mean scenario: expected daily demand equals the selected historical mean. This is not a selected-model forecast.")
input_mode = st.radio("Inventory inputs", ["Enter assumptions", "Synthetic example"], key="inventory_input_mode", horizontal=True)
synthetic = input_mode == "Synthetic example"
input_context = (dataset.source, dataset.provenance, fingerprint, store, family, input_mode)
defaults = {"current_inventory": 100. if synthetic else None, "lead_time_days": 7 if synthetic else None,
            "service_level_pct": 95., "ordering_cost": 50. if synthetic else None,
            "annual_holding_cost": 2. if synthetic else None, "outstanding_orders": 0. if synthetic else None,
            "currency": "LKR" if synthetic else "", "receipt_day": None, "allow_history_extension": False}
if st.session_state.get("inventory_input_context") != input_context:
    st.session_state["inventory_input_context"] = input_context
    st.session_state["inventory_plan"] = None
    for name, value in defaults.items():
        st.session_state[f"inventory_{name}"] = value
signature = (input_context, mode, horizon, forecast_signature)
if st.session_state.get("inventory_plan_signature") != signature:
    st.session_state["inventory_plan_signature"] = signature
    st.session_state["inventory_plan"] = None
st.caption("All stock/cost values below are synthetic example assumptions, even if edited." if synthetic else "Enter your scenario assumptions. The editable 95% target is only an assumption; no stock or costs are inferred from sales.")
with st.form("inventory_inputs_form"):
    left, right = st.columns(2)
    with left:
        on_hand = st.number_input("Current inventory (units)", min_value=0., value=None, step=1., key="inventory_current_inventory")
        lead_time = st.number_input("Supplier lead time (calendar days)", min_value=0, max_value=365, value=None, step=1, key="inventory_lead_time_days")
        service = st.number_input("Desired cycle service level (%)", min_value=50., max_value=99.9, value=None, step=.1, key="inventory_service_level_pct")
        outstanding = st.number_input("Outstanding purchase orders (units; enter 0 if none)", min_value=0., value=None, step=1., key="inventory_outstanding_orders")
    with right:
        currency = st.text_input("Currency label (three uppercase letters)", placeholder="e.g. LKR or USD", key="inventory_currency")
        ordering_cost = st.number_input("Ordering cost (currency per order)", min_value=0., value=None, step=1., key="inventory_ordering_cost")
        holding_cost = st.number_input("Annual holding cost (currency per unit per year)", min_value=0., value=None, step=.1, key="inventory_annual_holding_cost")
        receipt_day = st.number_input("Outstanding orders arrive on day (optional)", min_value=0, max_value=365, value=None, step=1, key="inventory_receipt_day")
    st.caption("Outstanding units are one aggregate receipt. Day 0 arrives before demand; later receipts arrive at the END of that day. Blank timing excludes receipts from shortage projections, but units still count in inventory position. Annual holding cost is an amount, not a percentage.")
    extension = st.checkbox("Allow historical-mean demand beyond the forecast window when lead time is longer", key="inventory_allow_history_extension")
    submitted = st.form_submit_button("Calculate inventory scenario", type="primary")
if submitted:
    st.session_state["inventory_plan"] = None
    try:
        inputs = InventoryInputs(on_hand, lead_time, None if service is None else service / 100, ordering_cost, holding_cost,
                                 outstanding, currency.strip(), receipt_day, extension,
                                 "Synthetic example assumptions" if synthetic else "User-entered scenario assumptions")
        st.session_state["inventory_plan"] = plan_inventory(history, inputs, source=dataset.source,
                                                            provenance=dataset.provenance, store=store, family=family,
                                                            future=future, forecast_horizon=horizon)
    except ValueError as exc:
        st.error(f"Scenario could not be calculated: {exc}")
plan = st.session_state.get("inventory_plan")
if plan is None:
    st.info("Complete the required inputs, then calculate the scenario. No inventory recommendation has been generated.")
    st.stop()
summary = plan.summary
st.subheader("Submitted inventory scenario")
st.caption("Results use the last submitted inputs. After editing the form, click Calculate inventory scenario to update them.")
st.write(f"Inventory inputs: {plan.inputs.input_provenance}. Demand basis: {plan.metadata['demand_basis']}.")
cards = st.columns(4)
cards[0].metric("Average daily demand", f"{summary['average_daily_demand']:,.2f}")
cards[1].metric("Safety stock (units)", f"{summary['safety_stock']:,.2f}")
cards[2].metric("Reorder point (units)", f"{summary['reorder_point']:,.2f}")
cards[3].metric("EOQ (units)", f"{summary['eoq']:,.2f}")
cards = st.columns(4)
cards[0].metric("Inventory position", f"{summary['inventory_position']:,.2f}")
cards[1].metric("Suggested replenishment units", f"{summary['recommended_quantity']:,}")
cards[2].metric("Days of stock (on-hand only)", "N/A — zero mean demand" if summary['days_of_stock_remaining'] is None else f"{summary['days_of_stock_remaining']:,.1f}")
cards[3].metric("Expected lead-time demand", f"{summary['lead_time_demand']:,.2f}")
st.caption("Safety stock uses historical daily sales variability, not calibrated forecast errors. It assumes independent demand, fixed lead time, and a normal approximation. Cycle service is not guaranteed fill rate. EOQ annualizes the selected mean over 365 days; seasonality can make this misleading.")
if summary['stockout_risk']:
    st.warning("Potential shortage — receipt timing unknown." if summary['receipt_timing_unknown'] else "Projected shortage before replenishment.")
else:
    st.info(f"No projected unmet demand during the {plan.inputs.lead_time_days}-day lead-time window under these assumptions. This is not a guarantee.")
if summary['overstock_risk']:
    st.warning("Potential excess inventory: position is above reorder point plus one EOQ.")
else:
    st.caption("The reorder-point-plus-EOQ excess threshold is not exceeded.")
for recommendation in plan.recommendations:
    st.write(f"• {recommendation}")
with st.expander("Submitted assumptions and calculation audit"):
    st.dataframe(pd.DataFrame([asdict(plan.inputs)]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([summary]), hide_index=True, width="stretch")
    st.json(plan.metadata)
    st.write("If position is at/below reorder point, quantity = ceil(max(EOQ, reorder point − position, 0)); otherwise zero. Only that quantity is rounded up. No initial backorders, pack sizes, minimum orders, capacity limits, or quantity discounts are modelled.")
with st.expander("Lead-time stock projection (without a new order)"):
    st.caption("Demand is consumed before any end-of-day existing PO receipt. Unmet units are lost sales, not a carried backlog. Late orders can create both a near-term shortage and an inventory-position excess flag.")
    if plan.projection.empty:
        st.info("Zero lead time: no waiting-day demand is projected; receipt is assumed immediate.")
    else:
        st.dataframe(plan.projection, hide_index=True, width="stretch")
st.download_button("Download inventory scenario CSV", plan.export_frame().to_csv(index=False, lineterminator="\n").encode("utf-8"),
                   file_name=f"inventory_scenario_store_{store}_{end:%Y%m%d}.csv", mime="text/csv")
