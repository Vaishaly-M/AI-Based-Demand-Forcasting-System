"""Smoke-check the entry point and each registered navigation destination."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("page", [
    "overview", "sales_analytics", "demand_forecasting", "model_evaluation",
    "inventory_planning", "business_insights",
])
def test_navigation_pages_render(page: str) -> None:
    """Navigation destinations render without runtime exceptions."""
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint).run()
    assert not app.exception
    if page != "overview":
        app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception
    assert len(app.title) >= 1
