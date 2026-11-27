"""Transparent family-level inventory scenarios, not executable SKU orders."""

from dataclasses import asdict, dataclass
from math import ceil, isclose, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.demand_planning.data.time_series import validate_daily_series
from src.demand_planning.evaluation.experiment import series_fingerprint
from src.demand_planning.forecasting.future import FORECAST_VERSION, FutureForecast, select_forecast_horizon


@dataclass(frozen=True)
class InventoryInputs:
    """User assumptions in units, calendar days, and one declared currency.

    Outstanding orders are one aggregate receipt. Day 0 is before the first
    forecast day; later receipts arrive at the END of the specified day. None
    means unknown timing and excludes receipts from shortage projections.
    """

    current_inventory: float
    lead_time_days: int
    service_level: float
    ordering_cost: float
    annual_holding_cost: float
    outstanding_orders: float
    currency: str
    receipt_day: int | None = None
    allow_history_extension: bool = False
    input_provenance: str = "User-entered scenario assumptions"

    def validate(self) -> None:
        """Reject missing, nonfinite, negative, or unsupported assumptions."""
        for name in ("current_inventory", "ordering_cost", "annual_holding_cost", "outstanding_orders", "service_level"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if self.annual_holding_cost <= 0:
            raise ValueError("Annual holding cost must be positive, in currency per unit per year (not a percentage).")
        if not 0.50 <= self.service_level <= 0.999:
            raise ValueError("Cycle service level must be between 0.50 and 0.999; it is not a fill-rate guarantee.")
        for name in ("lead_time_days", "receipt_day"):
            value = getattr(self, name)
            if name == "receipt_day" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 365:
                raise ValueError(f"{name} must be a whole number of calendar days from 0 to 365.")
        if not isinstance(self.currency, str) or len(self.currency) != 3 or not self.currency.isascii() or not self.currency.isalpha() or self.currency != self.currency.upper():
            raise ValueError("Enter a three-letter uppercase currency label, such as LKR or USD; no currency conversion is performed.")
        if not isinstance(self.allow_history_extension, bool):
            raise ValueError("Historical extension must be explicitly enabled or disabled.")
        if self.input_provenance not in {"User-entered scenario assumptions", "Synthetic example assumptions"}:
            raise ValueError("Inventory input provenance must identify user assumptions or a synthetic example.")


@dataclass
class InventoryPlan:
    """One scenario with inspectable inputs, computed outputs, and risk reasoning."""

    inputs: InventoryInputs
    summary: dict
    projection: pd.DataFrame
    recommendations: list[str]
    metadata: dict

    def export_frame(self) -> pd.DataFrame:
        """Flatten the submitted scenario and its provenance into one CSV-ready row."""
        return pd.DataFrame([{**asdict(self.inputs), **self.summary, **self.metadata,
                              "recommendations": " | ".join(self.recommendations)}])


def _forecast_demand(history: pd.Series, future: FutureForecast, horizon: int, context: dict) -> pd.Series:
    """Verify forecast identity before using any session-held future result."""
    expected = {**context, "forecast_version": FORECAST_VERSION,
                "data_fingerprint": series_fingerprint(history), "trained_through": str(history.index[-1].date())}
    if any(future.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("No matching future forecast for the current data and filters. Generate a selected-model forecast first, or explicitly choose a historical-mean scenario.")
    result = select_forecast_horizon(future, horizon)
    rows = result.forecasts
    expected_dates = pd.date_range(history.index[-1], periods=horizon + 1, freq="D")[1:]
    if (rows.date.tolist() != expected_dates.tolist() or
            not rows.forecast_origin.eq(history.index[-1]).all() or
            future.model.trained_through != history.index[-1] or
            not rows.model.eq(future.model.name).all()):
        raise ValueError("Forecast dates, cutoff, or model do not match the planning context.")
    demand = validate_daily_series(pd.Series(rows.forecast.to_numpy(), index=pd.DatetimeIndex(rows.date)))
    if demand.isna().any():
        raise ValueError("Forecast demand cannot contain missing values.")
    return demand


def plan_inventory(
    history: pd.Series, inputs: InventoryInputs, *, source: str, provenance: str,
    store: int | None, family: str | None, future: FutureForecast | None = None,
    forecast_horizon: int = 30,
) -> InventoryPlan:
    """Calculate an explicit historical-mean or matching-forecast scenario.

    Safety stock uses sample SD of the full selected historical daily demand
    (ddof=1), NOT calibrated forecast errors. This assumes independent demand,
    fixed lead time and a normal approximation; cycle service is not guaranteed.
    No existing backorders, minimum order sizes, or pack constraints are assumed.
    """
    inputs.validate()
    sales = validate_daily_series(history)
    if len(sales) < 2 or sales.isna().any():
        raise ValueError("Inventory scenarios require at least two complete historical days to estimate variability. Missing demand is not imputed.")
    context = {"source": source, "provenance": provenance, "store": store, "family": family}
    history_mean = float(sales.mean())
    sigma = float(sales.std(ddof=1))
    lead_time = int(inputs.lead_time_days)
    extension_days = 0
    if future is not None:
        profile = _forecast_demand(history, future, forecast_horizon, context)
        average = float(profile.mean())
        basis = f"{len(profile)}-day selected-model forecast"
        model_name = future.model.name
        if lead_time > len(profile):
            extension_days = lead_time - len(profile)
            if not inputs.allow_history_extension:
                raise ValueError("Lead time exceeds the forecast window. Choose a longer forecast or explicitly enable historical-mean extension; demand is never silently truncated.")
        demand = np.concatenate([profile.to_numpy()[:lead_time], np.full(extension_days, history_mean)])
    else:
        average = history_mean
        basis = "Historical mean scenario (not a model forecast)"
        model_name = "Historical mean"
        demand = np.full(lead_time, history_mean)
    annual_demand = average * 365
    lead_demand = float(demand.sum())
    z_score = NormalDist().inv_cdf(float(inputs.service_level))
    safety_stock = z_score * sigma * sqrt(lead_time)
    reorder_point = lead_demand + safety_stock
    eoq = sqrt(2 * annual_demand * inputs.ordering_cost / inputs.annual_holding_cost)
    position = inputs.current_inventory + inputs.outstanding_orders
    threshold = reorder_point + eoq
    days_remaining = inputs.current_inventory / average if average > 0 else None
    calculated = (average, history_mean, sigma, annual_demand, lead_demand, safety_stock, reorder_point, eoq, position, threshold)
    if not all(isfinite(value) for value in calculated) or (days_remaining is not None and not isfinite(days_remaining)):
        raise ValueError("Inputs produce nonfinite inventory calculations. Check demand and cost magnitudes.")
    reorder_triggered = position <= reorder_point
    quantity = ceil(max(eoq, reorder_point - position, 0.)) if reorder_triggered else 0

    balance = float(inputs.current_inventory)
    if inputs.outstanding_orders > 0 and inputs.receipt_day == 0:
        balance += inputs.outstanding_orders
    rows = []
    for day, expected_demand in enumerate(demand, 1):
        opening = balance
        shortage = 0. if isclose(opening, float(expected_demand), rel_tol=1e-12, abs_tol=0.) else max(0., float(expected_demand) - opening)
        receipt = float(inputs.outstanding_orders) if inputs.receipt_day == day else 0.
        balance = max(0., opening - float(expected_demand)) + receipt
        rows.append({"lead_day": day, "opening_inventory": opening, "expected_demand": float(expected_demand),
                     "unmet_demand": shortage, "receipt_end_of_day": receipt, "closing_inventory": balance})
    projection = pd.DataFrame(rows, columns=["lead_day", "opening_inventory", "expected_demand", "unmet_demand", "receipt_end_of_day", "closing_inventory"])
    projection.insert(0, "date", pd.date_range(sales.index[-1], periods=lead_time + 1, freq="D")[1:])
    shortage_rows = projection.loc[projection.unmet_demand.gt(0)]
    first_shortage = int(shortage_rows.lead_day.iloc[0]) if len(shortage_rows) else None
    stockout_risk = first_shortage is not None
    overstock_risk = position > threshold
    unknown_receipt = inputs.outstanding_orders > 0 and inputs.receipt_day is None
    recommendations = []
    if stockout_risk:
        recommendations.append(f"Projected unmet demand begins on lead day {first_shortage}. Review expediting, transfers, or supplier timing; a normal new order arrives too late to prevent this shortage.")
    if unknown_receipt:
        recommendations.append("Confirm outstanding-order receipt timing. Those units count in inventory position but are excluded from the shortage projection until timing is supplied.")
    if quantity > 0:
        recommendations.append(f"Policy suggests {quantity:,} replenishment units because inventory position is at/below the reorder point. Verify SKU quantities, costs, and pack sizes before purchasing.")
    else:
        recommendations.append("No additional quantity is suggested by this reorder-point/EOQ rule; review any shortage or late-receipt alert separately.")
    if overstock_risk:
        recommendations.append("Inventory position exceeds reorder point plus one EOQ. Review outstanding orders or defer purchases; this is a heuristic excess indicator, not an optimized stock limit.")
    if average == 0:
        recommendations.append("Selected mean demand is zero: annualized demand and EOQ are zero; days of stock is undefined. Review the demand assumption and any historical-variability buffer before ordering.")
    if inputs.ordering_cost == 0:
        recommendations.append("Zero ordering cost gives EOQ zero in the continuous formula; there is no positive economic batch size. Only the reorder-point gap can generate a quantity.")
    if extension_days:
        recommendations.append(f"Lead-time demand includes {extension_days} days extended using the selected historical mean, not model predictions.")
    summary = {"average_daily_demand": average, "annualized_demand": annual_demand,
               "historical_daily_std": sigma, "z_score": z_score, "lead_time_demand": lead_demand,
               "safety_stock": safety_stock, "reorder_point": reorder_point, "eoq": eoq,
               "inventory_position": position, "recommended_quantity": quantity,
               "days_of_stock_remaining": days_remaining, "reorder_triggered": bool(reorder_triggered),
               "stockout_risk": stockout_risk, "overstock_risk": bool(overstock_risk),
               "overstock_threshold": threshold, "first_shortage_day": first_shortage,
               "projected_unmet_demand": float(projection.unmet_demand.sum()),
               "receipt_timing_unknown": bool(unknown_receipt), "extension_days": extension_days}
    metadata = {**context, "scenario_date": str(sales.index[-1].date()),
                "history_start": str(sales.index[0].date()), "history_days": len(sales),
                "demand_basis": basis, "model": model_name, "historical_mean": history_mean,
                "forecast_horizon_days": len(profile) if future is not None else None,
                "history_fingerprint": series_fingerprint(history),
                "variability_basis": "Selected historical daily sales sample SD (ddof=1); not calibrated forecast error",
                "service_assumption": "Cycle service target, not guaranteed fill rate; independent demand, fixed lead time, normal approximation",
                "receipt_policy": "One aggregate existing PO receipt; day 0 before demand, later days after demand; unknown timing excluded",
                "projection_policy": "No new recommended order included; unmet demand counted as lost sales, no carried backlog",
                "replenishment_policy": "If position <= reorder point: ceil(max(EOQ, reorder point - position, 0)); else 0",
                "rounding_policy": "Unrounded safety stock/ROP/EOQ for calculations; only recommended quantity rounded up",
                "unit_scope": "Illustrative product-family units; no SKU, pack-size, capacity, or minimum-order constraints",
                "backorders_assumed": 0, "uncertainty_status": "No calibrated forecast or service-level coverage"}
    return InventoryPlan(inputs, summary, projection, recommendations, metadata)
