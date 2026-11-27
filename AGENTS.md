# Development instructions

## Scope and workflow
- Build this portfolio project in the phases defined in PROJECT_PLAN.md. Phase 6 is authorized; obtain the user's approval before starting Phase 7.
- Inspect existing code and preserve user changes. Do not commit, publish, or deploy without instruction.
- Keep dependencies limited to requirements.txt and the Python standard library. Target Python 3.11+.
- Keep business logic in src/demand_planning, UI in app_pages, and tests in tests. Use pathlib, type hints, descriptive names, and public-function docstrings.
- Validate inputs and show actionable error/empty states. Never expose credentials in logs or the UI.

## Data and modelling
- Label generated data as synthetic. Never invent accuracy, savings, or business outcomes.
- Forecast one selected store/family initially. Sales measure demand units, not revenue; observed sales may understate demand during stockouts.
- Split chronologically before fitting preprocessing. Lag and rolling features must use only information available at the forecast origin. Never use future actuals during recursive evaluation.
- Compare seasonal naive, Linear Regression, Random Forest, and XGBoost using the same forecast origins and horizons. Select on validation only; reserve test data for final evaluation.
- Document zero-denominator metric behavior and clip demand predictions at zero. Publish uncertainty only with a documented calibration and coverage procedure.
- Inventory, costs, lead times, and purchase orders are user assumptions, not fields inferred from Kaggle sales. State units, service-level definition, and limitations.

## Quality
- Test feature leakage, chronological evaluation, zero-demand metrics, and inventory edge cases when those modules are implemented.
- Run python -m pytest and relevant smoke checks before reporting a phase complete; report actual results and limitations.
- Keep raw/processed data, model binaries, secrets, environments, and generated reports out of Git. A small explicitly synthetic example is allowed.
- Keep README and PROJECT_PLAN aligned with implemented behavior. Avoid unnecessary infrastructure or speculative abstractions.
