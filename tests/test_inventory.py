"""Independent inventory arithmetic, timing, validation, and forecast identity."""

from dataclasses import replace
from io import StringIO
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.demand_planning.evaluation.experiment import series_fingerprint
from src.demand_planning.forecasting.future import FORECAST_VERSION, FutureForecast
from src.demand_planning.forecasting.models import FittedModel
from src.demand_planning.inventory.planning import InventoryInputs, plan_inventory

CONTEXT = {"source": "Inventory fixture", "provenance": "Synthetic sales", "store": 1, "family": "A"}


def history(values: tuple = (8., 10., 12.)) -> pd.Series:
    """Mean 10 and sample SD 2 for the default hand-worked example."""
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def inputs(**changes) -> InventoryInputs:
    """Known assumptions producing EOQ 50, SS 4, and ROP 44."""
    base = InventoryInputs(20., 4, NormalDist().cdf(1.), 1., 2.92, 5., "LKR")
    return replace(base, **changes)


def future_fixture(sales: pd.Series, value: float = 2.) -> FutureForecast:
    """Controlled forecast values test planning arithmetic, not model accuracy."""
    dates = pd.date_range(sales.index[-1], periods=31)[1:]
    rows = pd.DataFrame({"date": dates, "forecast": value, "forecast_origin": sales.index[-1],
                         "model": "Seasonal Naive", "horizon_days": 30})
    model = FittedModel("Seasonal Naive", None, (), sales.index[-1], len(sales))
    metadata = {**CONTEXT, "forecast_version": FORECAST_VERSION, "data_fingerprint": series_fingerprint(sales),
                "trained_through": str(sales.index[-1].date()), "horizon_days": 30}
    return FutureForecast(model, rows, sales.copy(), metadata)


def test_hand_calculated_inventory_values() -> None:
    plan = plan_inventory(history(), inputs(), **CONTEXT)
    result = plan.summary
    assert result["average_daily_demand"] == 10
    assert result["annualized_demand"] == 3650
    assert result["historical_daily_std"] == 2
    assert result["safety_stock"] == pytest.approx(4)
    assert result["lead_time_demand"] == 40
    assert result["reorder_point"] == pytest.approx(44)
    assert result["eoq"] == 50
    assert result["inventory_position"] == 25
    assert result["recommended_quantity"] == 50
    assert result["days_of_stock_remaining"] == 2
    assert result["first_shortage_day"] == 3
    assert result["projected_unmet_demand"] == 20
    assert result["receipt_timing_unknown"]
    assert not result["overstock_risk"]


def test_zero_demand_and_zero_lead_time() -> None:
    zero = plan_inventory(history((0., 0., 0.)), inputs(current_inventory=0, outstanding_orders=0), **CONTEXT)
    assert zero.summary["days_of_stock_remaining"] is None
    for key in ("safety_stock", "reorder_point", "eoq", "recommended_quantity", "projected_unmet_demand"):
        assert zero.summary[key] == 0
    assert not zero.summary["stockout_risk"]
    excess = plan_inventory(history((0., 0.)), inputs(current_inventory=1), **CONTEXT)
    assert excess.summary["overstock_risk"]
    instant = plan_inventory(history(), inputs(lead_time_days=0, current_inventory=0, outstanding_orders=0), **CONTEXT)
    assert instant.summary["safety_stock"] == 0
    assert instant.summary["lead_time_demand"] == 0
    assert instant.summary["reorder_point"] == 0
    assert instant.summary["recommended_quantity"] == 50
    assert instant.projection.empty
    assert not instant.summary["stockout_risk"]


def test_zero_order_cost_gap_rounding_and_threshold_edges() -> None:
    sales = history((10., 10.))
    plan = plan_inventory(sales, inputs(current_inventory=20.1, outstanding_orders=0, lead_time_days=3, ordering_cost=0), **CONTEXT)
    assert plan.summary["eoq"] == 0
    assert plan.summary["recommended_quantity"] == 10
    at_threshold = plan_inventory(sales, inputs(current_inventory=30, outstanding_orders=0, lead_time_days=3), **CONTEXT)
    assert at_threshold.summary["reorder_triggered"]
    assert at_threshold.summary["recommended_quantity"] == 50
    above = plan_inventory(sales, inputs(current_inventory=30.01, outstanding_orders=0, lead_time_days=3), **CONTEXT)
    assert not above.summary["reorder_triggered"]
    assert above.summary["recommended_quantity"] == 0
    excess_boundary = plan_inventory(sales, inputs(current_inventory=80, outstanding_orders=0, lead_time_days=3), **CONTEXT)
    assert not excess_boundary.summary["overstock_risk"]
    excess = plan_inventory(sales, inputs(current_inventory=80.01, outstanding_orders=0, lead_time_days=3), **CONTEXT)
    assert excess.summary["overstock_risk"]


def test_receipt_day_zero_and_end_of_day_are_distinct() -> None:
    sales = history((10., 10.))
    base = inputs(current_inventory=0, lead_time_days=3, outstanding_orders=30, ordering_cost=0)
    immediate = plan_inventory(sales, replace(base, receipt_day=0), **CONTEXT)
    assert not immediate.summary["stockout_risk"]
    assert immediate.projection.opening_inventory.tolist() == [30., 20., 10.]
    end_day = plan_inventory(sales, replace(base, receipt_day=1), **CONTEXT)
    assert end_day.summary["first_shortage_day"] == 1
    assert end_day.summary["projected_unmet_demand"] == 10
    assert end_day.projection.closing_inventory.tolist() == [30., 20., 10.]
    assert end_day.projection.unmet_demand.tolist() == [10., 0., 0.]
    unknown = plan_inventory(sales, base, **CONTEXT)
    assert unknown.summary["inventory_position"] == 30
    assert unknown.summary["projected_unmet_demand"] == 30
    assert unknown.summary["receipt_timing_unknown"]
    late = plan_inventory(sales, replace(base, outstanding_orders=100, receipt_day=4), **CONTEXT)
    assert late.summary["stockout_risk"] and late.summary["overstock_risk"]
    assert late.summary["recommended_quantity"] == 0


def test_fractional_inventory_does_not_create_floating_point_shortage() -> None:
    plan = plan_inventory(history((.1, .1)), inputs(current_inventory=.3, lead_time_days=3, outstanding_orders=0), **CONTEXT)
    assert not plan.summary["stockout_risk"]
    assert plan.summary["projected_unmet_demand"] == 0


@pytest.mark.parametrize("name,value", [
    ("current_inventory", -1), ("current_inventory", np.inf), ("current_inventory", None),
    ("outstanding_orders", -1), ("outstanding_orders", np.nan), ("ordering_cost", -1),
    ("annual_holding_cost", 0), ("annual_holding_cost", -1), ("annual_holding_cost", "2"),
    ("service_level", 0.49), ("service_level", 1.), ("service_level", None), ("service_level", True),
    ("lead_time_days", -1), ("lead_time_days", 1.5), ("lead_time_days", True), ("lead_time_days", 366),
    ("receipt_day", -1), ("receipt_day", .5), ("currency", ""), ("currency", "lkr"),
    ("allow_history_extension", "yes"), ("input_provenance", "Real stock from Kaggle"),
])
def test_invalid_assumptions_rejected(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        plan_inventory(history(), inputs(**{name: value}), **CONTEXT)


def test_service_boundaries_and_missing_history() -> None:
    low = plan_inventory(history(), inputs(service_level=.5), **CONTEXT)
    high = plan_inventory(history(), inputs(service_level=.999), **CONTEXT)
    assert low.summary["safety_stock"] == 0
    assert high.summary["safety_stock"] > 12
    for sales in (history((10.,)), history((10., np.nan, 12.))):
        with pytest.raises(ValueError, match="two complete"):
            plan_inventory(sales, inputs(), **CONTEXT)


def test_forecast_lead_demand_and_explicit_extension() -> None:
    sales = history((8, 10, 12))  # integer values must retain fingerprint identity
    future = future_fixture(sales)
    plan = plan_inventory(sales, inputs(lead_time_days=3), future=future, forecast_horizon=7, **CONTEXT)
    assert plan.summary["average_daily_demand"] == 2
    assert plan.summary["lead_time_demand"] == 6
    assert plan.summary["historical_daily_std"] == 2
    assert plan.summary["annualized_demand"] == 730
    with pytest.raises(ValueError, match="exceeds"):
        plan_inventory(sales, inputs(lead_time_days=35), future=future, **CONTEXT)
    extended = plan_inventory(sales, inputs(lead_time_days=35, allow_history_extension=True), future=future, **CONTEXT)
    assert extended.summary["extension_days"] == 5
    assert extended.summary["lead_time_demand"] == 110  # 30*2 + 5*10
    assert extended.summary["average_daily_demand"] == 2  # EOQ basis stays selected forecast mean
    assert extended.projection.expected_demand.iloc[-5:].eq(10).all()


def test_stale_and_malformed_forecasts_rejected() -> None:
    sales = history()
    future = future_fixture(sales)
    with pytest.raises(ValueError, match="matching future"):
        plan_inventory(sales, inputs(), future=future, **{**CONTEXT, "family": "B"})
    changed = sales.copy()
    changed.iloc[0] = 999
    with pytest.raises(ValueError, match="matching future"):
        plan_inventory(changed, inputs(), future=future, **CONTEXT)
    future.forecasts.loc[0, "forecast_origin"] = pd.Timestamp("2020-01-01")
    with pytest.raises(ValueError, match="cutoff"):
        plan_inventory(sales, inputs(), future=future, **CONTEXT)
    future = future_fixture(sales)
    future.forecasts.loc[0, "forecast"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        plan_inventory(sales, inputs(), future=future, **CONTEXT)


def test_export_carries_inputs_provenance_and_policy() -> None:
    plan = plan_inventory(history(), inputs(input_provenance="Synthetic example assumptions"), **CONTEXT)
    exported = pd.read_csv(StringIO(plan.export_frame().to_csv(index=False)))
    assert exported.input_provenance.iloc[0] == "Synthetic example assumptions"
    assert exported.provenance.iloc[0] == "Synthetic sales"
    assert exported.currency.iloc[0] == "LKR"
    assert exported.recommended_quantity.iloc[0] == 50
    assert exported.scenario_date.iloc[0] == "2024-01-03"
    assert "not calibrated" in exported.variability_basis.iloc[0]
    assert exported.receipt_timing_unknown.iloc[0]
