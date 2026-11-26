# Architecture and development plan

## Scope

Build a portfolio-ready daily retail demand and inventory decision-support application for supply chain and analytics users. Forecast one selected store and product family initially. Only Phase 1 is currently authorized; stop for approval before Phase 2. Planned functionality must never appear operational before implementation.

## Architecture

```text
CSV -> validation/cleaning -> daily store/family series -> sales analytics
                            -> chronological splits -> features -> models
                               -> validation selection -> final test
                               -> future forecasts
Forecasts + user inventory assumptions -> inventory rules -> insights/CSV
Streamlit pages present results using KPI cards, Plotly charts, and tables.
```

Keep UI in `app_pages/` with common navigation in `streamlit_app.py`. Business logic lives in `src/demand_planning/`: data, features, forecasting, evaluation, inventory, and insights. Empty packages reserve boundaries; implement them in their assigned phases. Scripts provide reproducible CLI operations; tests cover domain behavior and UI smoke checks. Data, model binaries, and generated reports remain local and ignored. Use Joblib only for trusted local artifacts. No database or separate backend is required.

Engineering rules: Python 3.11+, pathlib paths, type hints, docstrings, reusable functions, input validation, descriptive names, separation of concerns, minimal dependencies. Keep secrets, environments, large datasets, and trained models out of Git. Preserve user changes. No fabricated accuracy, savings, or performance improvements.

## Evaluation methodology

- Data grain is daily date/store/family. README defines the initial schema and download procedure. Document cleaning, duplicates, missing dates, invalid values, and provenance. Observed sales are units, not revenue or necessarily unconstrained demand.
- Choose chronological train/validation/test dates after examining series lengths in Phase 2. Reserve adequate history for lag windows and 30-day horizons; never randomly shuffle time series.
- Fit learned preprocessing only on training folds. Shift target-based rolling windows and lags to exclude the target and future actuals. Calendar information is known in advance; promotions require explicit future availability.
- Start with weekly seasonal naive (period 7). Compare Linear Regression, Random Forest, and XGBoost on identical origins/horizons. Use rolling-origin cross-validation within the pre-test period where sufficient data exist.
- Multi-step backtests must reproduce production forecasting: recursive target features use earlier predictions, not unseen actuals. Clip negative predictions consistently before evaluation and export.
- Report MAE, RMSE, and WAPE. WAPE = `100 * sum(abs(actual - predicted)) / sum(abs(actual))`; zero denominator is undefined/N/A. Select by validation WAPE with MAE fallback for all-zero validation demand; document tie-breaking before test evaluation.
- Evaluate the selected model once on untouched test data, with no test-based tuning. Distinguish this result from any subsequent full-history refit for future forecasts. Record dates, seeds, parameters, versions, and feature definitions.
- Only show uncertainty with documented calibration, nominal coverage, measured holdout coverage, and sufficient residual samples. Consider horizon-specific rolling-validation residuals; no arbitrary percentage bands or guaranteed service claims.

## Inventory assumptions proposed for Phase 6

Require nonnegative current inventory and outstanding purchase orders; supplier lead time in days; desired cycle service level strictly between zero and one (initial UI range 0.50–0.999); ordering cost per order; and annual holding cost per unit per year (positive for EOQ), using a declared currency. These are user assumptions, never inferred inventory facts from Kaggle.

- Mean daily demand: explicitly documented forecast/historical window; annualized demand: daily mean × 365, with seasonality limitations.
- Safety stock: provisional `z * sigma_daily * sqrt(lead_time)`, with fixed lead time and independent daily errors. The service level is cycle service, not fill rate. Prefer calibrated lead-time errors when available; explain approximation limits for autocorrelation, intermittent demand, or variable lead times.
- Reorder point: expected lead-time demand plus safety stock. Disclose any historical-mean extension beyond forecast coverage; never silently truncate the lead time.
- EOQ: `sqrt(2 * annual_demand * ordering_cost / annual_holding_cost)` under steady demand, immediate replenishment, and no quantity discounts. Handle zero demand/cost explicitly.
- Inventory position: on-hand plus outstanding orders, initially assuming no backorders. Add a validated backorder input if that scope expands.
- If position is at/below reorder point, proposed replenishment is the rounded-up maximum of EOQ and the gap to reorder point, bounded below by zero. Explain policy and rounding.
- Days remaining: on-hand / daily demand; zero demand yields undefined/unbounded coverage rather than a division error.
- Stockout indicator: on-hand cannot cover expected demand before receipt. Aggregate outstanding orders do not establish receipt dates; request timing or clearly label this simplification.
- Overstock indicator: initially compare position with reorder point plus one EOQ; label it a heuristic, not an optimal business threshold.

Family-level inventory is an educational scenario because families combine heterogeneous items. Production orders need SKU-level demand/units/costs, supplier calendars, pack sizes, minimum quantities, and receipt timing. Synthetic assumptions must be visibly labelled.

## Development phases and acceptance criteria

### Phase 1 — Initialization (current)

Status: complete. Generator and six-page AppTest suite: 17 passing tests on Python 3.12.6. Dependency and syntax checks passed. Awaiting approval for Phase 2; no analytics or model training has begun.

Deliver structure, AGENTS.md, this plan, dependency/Git configuration, README, navigation placeholders, and synthetic generator.

Acceptance: PowerShell setup documented; dashboard runs without external data; all six pages render; seeded generator produces unique complete daily panels with nonnegative sales and explicit synthetic flags; invalid inputs fail; existing output is protected; data and environments are ignored. Execute tests/checks and disclose limits. Stop for approval.

### Phase 2 — Data preparation and exploratory analysis

Implement CSV loading, validation, cleaning report, daily series construction, and store/family filters.

Acceptance: handle invalid schemas, empty inputs, duplicates, missing dates, nonfinite/negative values, and insufficient history explicitly. Display total sales units, average daily demand, growth with documented comparison windows/zero-base handling, daily/weekly/monthly trends, seasonal patterns, and high-/low-demand family rankings. Distinguish incomplete periods and provide loading/empty/error states. Record data provenance.

### Phase 3 — Features and baseline

Implement calendar, lag, rolling-window features, and weekly seasonal naive forecasts.

Acceptance: tests prove no target/future leakage and cover short histories, rolling boundaries, and baseline recursion. Forecast 7, 14, and 30 days with nonnegative outputs; document temporal split boundaries.

### Phase 4 — Training and evaluation

Implement Linear Regression, Random Forest, and XGBoost with seeded runs, training-only preprocessing, rolling-origin comparisons, and validation-based selection.

Acceptance: common origins/horizons, no shuffled splits, metrics tests including all-zero demand, final untouched test evaluation, actual-versus-predicted outputs, and model comparison from real runs. Persist trusted artifacts with metadata using Joblib; no fabricated results.

### Phase 5 — Future forecasts

Implement selected store/family 7/14/30-day forecasting and recursive feature updates.

Acceptance: contiguous future dates, no future actual dependence, finite nonnegative outputs, explicit training cutoff, interactive Plotly history/forecast chart, and export-ready tables. Intervals require documented calibration and coverage; omit when unsupported. Test multi-step recursion and invalid horizons.

### Phase 6 — Inventory planning

Implement validated inputs, average demand, safety stock, reorder point, EOQ, inventory position, replenishment quantity, and days remaining.

Acceptance: hand-calculated unit tests plus zero demand, invalid costs/service levels, zero lead time, outstanding orders, and rounding edges. Clearly explain units, assumptions, family-level limitations, stockout/overstock heuristics, and recommended actions.

### Phase 7 — Dashboard integration and business insights

Connect modules with sidebar filters/navigation, consistent minimal theme, metric cards, Plotly charts, readable tables, and plain-language guidance.

Acceptance: complete data-to-recommendation flow; demand increases/decreases with explicit comparison periods and zero-baseline handling; replenishment, shortage, and excess indicators; rule-based procurement recommendations. Forecast/recommendation CSV exports include dates, selections, units, model, provenance, and assumptions. Loading, empty, invalid, and insufficient-history states are actionable. Avoid decorative animations and implementation details in business flows.

### Phase 8 — Testing, documentation, and deployment readiness

Complete regression tests, reproducible setup, example screenshots, methodology, limitations, and minimal Streamlit hosting instructions.

Acceptance: full suite passes in a clean environment; record exact tested dependencies; no datasets, models, secrets, or environments tracked; README reflects actual results and limitations. Document resource limits and deployment configuration. Actual publication requires an explicit instruction.

## Open decisions

- Real Kaggle data have not been inspected; download is unnecessary for Phase 1.
- Split dates, lags, fold counts, tuning budgets, and interval feasibility depend on Phase 2 findings.
- Inventory receipt timing, family unit interpretation, and cost units need explicit UX decisions in Phase 6.
- Dependency ranges are not a lock file; capture exact tested versions before final portfolio release.
