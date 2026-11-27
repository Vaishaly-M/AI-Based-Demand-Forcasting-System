"""Smoke-check the entry point and each registered navigation destination."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.demand_planning.data.loading import SalesDataError, validate_sales


@pytest.mark.parametrize("page", [
    "overview", "sales_analytics", "demand_forecasting", "model_evaluation",
    "inventory_planning", "business_insights",
])
def test_navigation_pages_render(page: str) -> None:
    """Navigation destinations render without runtime exceptions."""
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()
    assert not app.exception
    if page != "overview":
        app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception
    assert len(app.title) >= 1


def test_analytics_filters_and_empty_source() -> None:
    """Changing filters rerenders analytics; missing upload cannot retain demo data."""
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()
    app.switch_page("app_pages/sales_analytics.py").run()
    assert not app.exception
    assert any(metric.label == "Observed sales units" for metric in app.metric)
    app.sidebar.selectbox(key="selected_store").set_value(2).run()
    app.sidebar.selectbox(key="selected_family").set_value("PRODUCE").run()
    assert not app.exception
    assert any("Store 2" in heading.value and "PRODUCE" in heading.value for heading in app.subheader)
    app.radio[0].set_value("Monthly").run()
    assert not app.exception
    app.sidebar.selectbox(key="data_source").set_value("Upload CSV").run()
    assert not app.exception
    assert app.session_state["sales_dataset"] is None
    assert any("No valid sales" in message.value for message in app.info)


def test_single_day_and_gap_only_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal histories and ranges containing only missing dates render safely."""
    from app_pages import data_controls

    dataset = validate_sales(pd.DataFrame({
        "date": ["2024-01-01", "2024-01-03"], "store_nbr": [1, 1],
        "family": ["A", "A"], "sales": [0., 2.],
    }))
    monkeypatch.setattr(data_controls, "demo_dataset", lambda: dataset)
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()
    app.switch_page("app_pages/sales_analytics.py").run()
    assert not app.exception
    assert any("no observation" in message.value for message in app.warning)
    date = pd.Timestamp("2024-01-02").date()
    app.sidebar.date_input(key="sales_dates").set_value((date, date)).run()
    assert not app.exception
    assert any("No observed sales" in message.value for message in app.info)
    date = pd.Timestamp("2024-01-01").date()
    app.sidebar.date_input(key="sales_dates").set_value((date, date)).run()
    assert not app.exception
    assert any("Fewer than 14" in message.value for message in app.info)


def test_validation_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rejected data is shown as an error, never a traceback or stale dashboard."""
    from app_pages import data_controls

    def invalid_input() -> None:
        raise SalesDataError("sales: invalid negative values; correct the input CSV.")

    monkeypatch.setattr(data_controls, "demo_dataset", invalid_input)
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()
    app.switch_page("app_pages/sales_analytics.py").run()
    assert not app.exception
    assert any("correct the input" in message.value for message in app.error)
    assert app.session_state["sales_dataset"] is None


def test_baseline_horizons_cutoff_and_short_history() -> None:
    """Forecast page obeys selected cutoff and has an actionable short-history state."""
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=15).run()
    app.switch_page("app_pages/demand_forecasting.py").run()
    app.radio(key="forecast_method").set_value("Seasonal Naive preview").run()
    for horizon in (7, 14, 30):
        app.selectbox(key="forecast_horizon").set_value(horizon).run()
        assert not app.exception
        table = app.dataframe[0].value
        assert len(table) == horizon
        assert table["Forecast units"].ge(0).all()
    start = pd.Timestamp("2024-01-01").date()
    end = pd.Timestamp("2024-01-14").date()
    app.sidebar.date_input(key="sales_dates").set_value((start, end)).run()
    assert not app.exception
    assert app.dataframe[0].value["Date"].min() == pd.Timestamp("2024-01-15")
    app.sidebar.date_input(key="sales_dates").set_value((start, start)).run()
    assert not app.exception
    assert any("seven consecutive" in message.value for message in app.warning)
    assert len(app.dataframe) == 0
    app.sidebar.selectbox(key="data_source").set_value("Upload CSV").run()
    assert not app.exception
    assert any("Choose a valid sales" in message.value for message in app.info)


def test_model_evaluation_requires_click_and_clears_changed_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual pipeline via UI, then ensure another family has no stale result."""
    from app_pages import data_controls
    from src.demand_planning.data.synthetic import generate_synthetic_sales

    dataset = validate_sales(generate_synthetic_sales(days=116), "Synthetic UI fixture")
    monkeypatch.setattr(data_controls, "demo_dataset", lambda: dataset)
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    app.switch_page("app_pages/model_evaluation.py").run()
    assert not app.exception
    assert app.session_state["evaluation_result"] is None
    next(button for button in app.button if button.label == "Run model evaluation").click().run()
    assert not app.exception
    result = app.session_state["evaluation_result"]
    assert result is not None
    assert any("Selected on validation" in message.value for message in app.success)
    app.radio[0].set_value("test").run()
    assert not app.exception
    assert app.session_state["evaluation_result"].metadata == result.metadata
    app.sidebar.selectbox(key="selected_family").set_value("PRODUCE").run()
    assert not app.exception
    assert app.session_state["evaluation_result"] is None
    assert any(button.label == "Run model evaluation" for button in app.button)


def test_selected_future_workflow_and_stale_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluate, refit, switch horizons, and reject a different family's selection."""
    from app_pages import data_controls
    from src.demand_planning.data.synthetic import generate_synthetic_sales

    dataset = validate_sales(generate_synthetic_sales(days=116), "Synthetic future UI fixture")
    monkeypatch.setattr(data_controls, "demo_dataset", lambda: dataset)
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    app.switch_page("app_pages/demand_forecasting.py").run()
    assert not app.exception
    assert any("Run Model Evaluation" in message.value for message in app.info)
    app.switch_page("app_pages/model_evaluation.py").run()
    next(button for button in app.button if button.label == "Run model evaluation").click().run()
    assert not app.exception
    earlier_cutoff = app.session_state["evaluation_result"].selected_model.trained_through
    app.switch_page("app_pages/demand_forecasting.py").run()
    assert not app.exception
    next(button for button in app.button if button.label == "Refit selected model and forecast").click().run()
    assert not app.exception
    future = app.session_state["future_result"]
    assert future.model.trained_through > earlier_cutoff
    for horizon in (7, 14, 30):
        app.selectbox(key="forecast_horizon").set_value(horizon).run()
        assert not app.exception
        assert len(app.dataframe[0].value) == horizon
        assert app.dataframe[0].value["Date"].min() > pd.Timestamp(dataset.frame.date.max())
    assert app.session_state["evaluation_result"].selected_model.trained_through == earlier_cutoff
    app.switch_page("app_pages/inventory_planning.py").run()
    assert not app.exception
    app.radio(key="inventory_input_mode").set_value("Synthetic example").run()
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    assert app.session_state["inventory_plan"].metadata["model"] == future.model.name
    assert app.session_state["inventory_plan"].summary["average_daily_demand"] == pytest.approx(future.forecasts.forecast.mean())
    app.number_input(key="inventory_lead_time_days").set_value(31)
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    assert app.session_state["inventory_plan"] is None
    assert any("exceeds" in message.value for message in app.error)
    app.checkbox(key="inventory_allow_history_extension").check()
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    assert app.session_state["inventory_plan"].summary["extension_days"] == 1
    app.sidebar.selectbox(key="selected_family").set_value("PRODUCE").run()
    assert app.session_state["inventory_plan"] is None
    app.switch_page("app_pages/demand_forecasting.py").run()
    assert not app.exception
    assert app.session_state["future_result"] is None
    assert not app.dataframe
    assert any("matching evaluation" in message.value for message in app.info)


def test_inventory_manual_validation_and_synthetic_scenario() -> None:
    """No implied inventory values; submitted scenarios clear on invalid input or filter changes."""
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=30).run()
    app.switch_page("app_pages/inventory_planning.py").run()
    assert not app.exception
    assert any("matching selected-model forecast" in message.value for message in app.info)
    app.radio(key="inventory_demand_basis").set_value("Historical mean scenario").run()
    assert not app.exception
    assert app.number_input(key="inventory_current_inventory").value is None
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["inventory_plan"] is None
    app.radio(key="inventory_input_mode").set_value("Synthetic example").run()
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    plan = app.session_state["inventory_plan"]
    assert plan.inputs.input_provenance == "Synthetic example assumptions"
    assert plan.summary["recommended_quantity"] >= 0
    assert len(app.metric) == 8
    app.number_input(key="inventory_annual_holding_cost").set_value(0.)
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    assert not app.exception
    assert app.session_state["inventory_plan"] is None
    assert any("holding cost" in message.value for message in app.error)
    app.number_input(key="inventory_annual_holding_cost").set_value(2.)
    next(button for button in app.button if button.label == "Calculate inventory scenario").click().run()
    app.sidebar.selectbox(key="selected_family").set_value("PRODUCE").run()
    assert not app.exception
    assert app.session_state["inventory_plan"] is None
