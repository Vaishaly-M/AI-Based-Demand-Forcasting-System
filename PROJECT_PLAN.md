# Architecture and development plan

## Scope

Build a portfolio-ready daily retail demand and inventory decision-support application for supply chain and analytics users. Forecast one selected store and product family initially. Phase 6 was authorized and is now complete; stop for approval before Phase 7. Planned functionality must never appear operational before implementation.

## Architecture

```text
CSV -> validation/cleaning -> daily store/family series -> sales analytics
                            -> chronological splits -> features -> models
                               -> validation selection -> final test
                               -> future forecasts
Forecasts + user inventory assumptions -> inventory rules -> insights/CSV
Streamlit pages present results using KPI cards, Plotly charts, and tables.
```

Keep UI in `app_pages/` with common navigation in `streamlit_app.py`. Business logic lives in `src/demand_planning/`: data, analytics, features, forecasting, evaluation, inventory, and insights. Empty packages reserve later-phase boundaries. Scripts provide reproducible CLI operations; tests cover domain behavior and UI smoke checks. Data, model binaries, and generated reports remain local and ignored. Use Joblib only for trusted local artifacts. No database or separate backend is required.

Engineering rules: Python 3.11+, pathlib paths, type hints, docstrings, reusable functions, input validation, descriptive names, separation of concerns, minimal dependencies. Keep secrets, environments, large datasets, and trained models out of Git. Preserve user changes. No fabricated accuracy, savings, or performance improvements.

## Evaluation methodology

- Data grain is daily date/store/family. README defines the initial schema and download procedure. Document cleaning, duplicates, missing dates, invalid values, and provenance. Observed sales are units, not revenue or necessarily unconstrained demand.
- Phase 3 split defaults reserve the final 30 daily observations for test, preceding 30 for validation, and at least 56 training days (28 warm-up + 28 usable rows). Require a complete daily calendar and targets; never shuffle. Freeze the chosen experiment's boundaries before Phase 4 selection; any changes must be justified without test results.
- Fit learned preprocessing only on training folds. Shift target-based rolling windows and lags to exclude the target and future actuals. Calendar information is known in advance; promotions require explicit future availability.
- Start with weekly seasonal naive (period 7). Compare Linear Regression, Random Forest, and XGBoost on identical origins/horizons. Use rolling-origin cross-validation within the pre-test period where sufficient data exist.
- Multi-step backtests must reproduce production forecasting: recursive target features use earlier predictions, not unseen actuals. Clip negative predictions consistently before evaluation and export.
- Report MAE, RMSE, and WAPE. WAPE = `100 * sum(abs(actual - predicted)) / sum(abs(actual))`; zero denominator is undefined/N/A. Select by validation WAPE with MAE fallback for all-zero validation demand; document tie-breaking before test evaluation.
- Evaluate the selected model once on untouched test data, with no test-based tuning. Distinguish this result from any subsequent full-history refit for future forecasts. Record dates, seeds, parameters, versions, and feature definitions.
- Only show uncertainty with documented calibration, nominal coverage, measured holdout coverage, and sufficient residual samples. Consider horizon-specific rolling-validation residuals; no arbitrary percentage bands or guaranteed service claims.

## Inventory assumptions implemented in Phase 6

Require finite nonnegative current inventory, outstanding purchase orders, and ordering cost; whole-day lead time from 0–365; cycle service level 0.50–0.999; and positive annual holding cost per unit per year. Use one declared uppercase three-letter currency label, with no conversion. Optional aggregate PO receipt day is 0–365 or unknown. These are user assumptions or explicitly labelled synthetic examples, never inventory facts inferred from Kaggle.

- Mean daily demand: selected 7/14/30-day forecast mean or explicitly selected historical mean; annualized demand: mean × 365, with seasonality limitations. Historical variance always uses all selected complete history, with a mechanical minimum of two days.
- Safety stock: provisional `z * sigma_daily * sqrt(lead_time)`, using historical daily sales sample SD (ddof=1), not calibrated forecast error. Fixed lead time, independent demand, and a normal approximation are assumed. Cycle service is not guaranteed fill rate. Calibrated lead-time error estimation remains unsupported; disclose limitations for seasonality, autocorrelation, intermittent demand, and supplier variability.
- Reorder point: expected lead-time demand plus safety stock. Disclose any historical-mean extension beyond forecast coverage; never silently truncate the lead time.
- EOQ: `sqrt(2 * annual_demand * ordering_cost / annual_holding_cost)` under steady demand, full-lot receipt, and no quantity discounts. Zero mean demand or ordering cost yields zero EOQ; zero holding cost is rejected. With zero ordering cost, the reorder-point gap alone can suggest a quantity.
- Inventory position: on-hand plus outstanding orders, initially assuming no backorders. Add a validated backorder input if that scope expands.
- If position is at/below reorder point, proposed replenishment is the rounded-up maximum of EOQ and the gap to reorder point, bounded below by zero. Explain policy and rounding.
- Days remaining: on-hand / daily demand; zero demand yields undefined/unbounded coverage rather than a division error.
- Stockout indicator: daily unmet demand during the new-order lead time. Day-zero outstanding orders arrive before demand; later receipts arrive at day end. Unknown receipt timing is excluded from this projection but included in inventory position. Unmet demand is lost sales, not a carried backlog. The new suggested order is not included in the waiting-period projection. This is a conditional indicator, not a probability.
- Overstock indicator: initially compare position with reorder point plus one EOQ; label it a heuristic, not an optimal business threshold.

Family-level inventory is an educational scenario because families combine heterogeneous items. Production orders need SKU-level demand/units/costs, supplier calendars, pack sizes, minimum quantities, and receipt timing. Synthetic assumptions must be visibly labelled.

## Development phases and acceptance criteria

### Phase 1 — Initialization (complete)

Status: complete. At the Phase 1 checkpoint, generator and six-page AppTest suite: 17 passing tests on Python 3.12.6. Dependency and syntax checks passed. Phase 2 was subsequently authorized.

Deliver structure, AGENTS.md, this plan, dependency/Git configuration, README, navigation placeholders, and synthetic generator.

Acceptance: PowerShell setup documented; dashboard runs without external data; all six pages render; seeded generator produces unique complete daily panels with nonnegative sales and explicit synthetic flags; invalid inputs fail; existing output is protected; data and environments are ignored. Execute tests/checks and disclose limits. Stop for approval.

### Phase 2 — Data preparation and exploratory analysis

Status: complete. Added strict CSV loading, auditable normalization, per-series coverage, shared source/store/family/date controls, KPI cards, Plotly trends and seasonal profiles, and common-window family rankings. Test suite: 55 passing tests. The full Kaggle file has not been downloaded or benchmarked; the schema is exercised with local/upload fixtures and the reproducible demo.

Policy decisions: reject invalid/duplicate observations rather than silently drop/sum them; preserve missing days as unknown; observed-day KPI denominators; final seven days versus preceding seven for growth, requiring complete windows and positive prior total; calendar-boundary periods marked incomplete; incomplete families excluded from ranks. At the Phase 2 checkpoint, model features, training, evaluation splits, and inventory calculations were not implemented.

Demo findings: six series of 365 days, 2,190 rows, 35 observed zero-sales rows, no internal gaps, and a partial December ending 2024-12-30. Annual seasonal recurrence cannot be established with this one-year demonstration. Final model split dates, feature windows, and minimum training lengths remain Phase 3/4 decisions and must be chosen before test evaluation.

Implement CSV loading, validation, cleaning report, daily series construction, and store/family filters.

Acceptance: handle invalid schemas, empty inputs, duplicates, missing dates, nonfinite/negative values, and insufficient history explicitly. Display total sales units, average daily demand, growth with documented comparison windows/zero-base handling, daily/weekly/monthly trends, seasonal patterns, and high-/low-demand family rankings. Distinguish incomplete periods and provide loading/empty/error states. Record data provenance.

### Phase 3 — Features and baseline

Status: complete. Calendar fields and weekday cycles, lags 1/7/14/28, shifted rolling means/population standard deviations for 7/14/28 days, a next-day feature builder, chronological raw-series split contracts, and a weekly Seasonal Naive baseline are implemented. The forecast page previews 7/14/30-day baselines for the selected store/family and history cutoff. No ML training, scoring, interval calibration, or inventory calculations were added.

Acceptance evidence: 88 tests passed without warnings. Checks cover hand-calculated lags/rolling statistics, warm-up boundaries, gap propagation, invalid daily indexes, target/future-value perturbations, prediction-fed next-day features, weekly repetition beyond day seven, zero-demand outputs, split isolation, and UI horizons/cutoffs/short histories. The baseline requires a complete final week; features require 28 known days for a future row. No implicit gap filling.

Default demo split: train 2024-01-01–2024-10-31 (305 days), validation 2024-11-01–2024-11-30 (30), test 2024-12-01–2024-12-30 (30). Split functions enforce at least 116 complete daily observations overall. These are structural minima, not statistical adequacy claims. Historical feature matrices are one-step examples; multi-step holdouts must rebuild features from predictions, never later holdout actuals. No holdout metrics have been computed. The interactive baseline preview is separate from the formal evaluation workflow.

Implement calendar, lag, rolling-window features, and weekly seasonal naive forecasts.

Acceptance: tests prove no target/future leakage and cover short histories, rolling boundaries, and baseline recursion. Forecast 7, 14, and 30 days with nonnegative outputs; document temporal split boundaries.

### Phase 4 — Training and evaluation

Status: complete. Added MAE/RMSE/WAPE with zero-demand handling, training-only StandardScaler/Linear Regression, seeded Random Forest and XGBoost, recursive multi-step prediction for evaluation, expanding-window diagnostic folds, validation-only model selection, winner-only test assessment, actual/predicted charts, a reproducible CLI, and Joblib/JSON/CSV local artifact bundles. No full-history production refit or inventory work was added.

Frozen initial settings: Random Forest 100 trees/depth 10/min leaf 2; XGBoost 120 trees/depth 3/learning rate 0.05/full sampling/hist/squared error; seed 42 and one worker. No hyperparameter search or early stopping. Up to three 30-day diagnostic CV holdouts lie entirely inside training, 30 days apart, each preceded by at least 56 training days. Dedicated 30-day validation WAPE selects the model; all-zero demand falls back to MAE. Ties use MAE, RMSE, then Seasonal Naive, Linear Regression, Random Forest, XGBoost. Refitting uses train plus validation, excludes test, and tests only the winner. Prefix horizons 7/14/30 are reported, not treated as independent trials.

Verification: 106 tests passed without warnings. Tests cover metrics, scaler isolation, actual estimators, reproducibility, recursive clipping, selection/ties, unchanged predictions/selection after test-target perturbation, artifact round trips, and UI run/invalidation behavior. The default synthetic Store 1 / DAIRY CLI run selected Seasonal Naive (30-day validation WAPE 16.08%); its reserved-test WAPE was 29.91%, MAE 14.3667, RMSE 19.4414. These are observed synthetic results, not real-data accuracy claims. Final local audit bundle: reports/evaluation/phase4-demo-final (ignored by Git). README records all four validation scores and exact methodology.

The UI cannot prevent a human from reusing viewed test data to revise an experiment; it explains that limitation. Fixed settings were not revised based on test performance. Saved models are fitted only through validation end. Phase 5 must distinguish any future full-history refit from these evaluation results. No calibrated intervals have been produced.

Implement Linear Regression, Random Forest, and XGBoost with seeded runs, training-only preprocessing, rolling-origin comparisons, and validation-based selection.

Acceptance: common origins/horizons, no shuffled splits, metrics tests including all-zero demand, final untouched test evaluation, actual-versus-predicted outputs, and model comparison from real runs. Persist trusted artifacts with metadata using Joblib; no fabricated results.

### Phase 5 — Future forecasts

Status: complete. Added a separate full-selected-history refit of the validation winner, exact evaluation-context checks, supported 7/14/30-day recursive future forecasts, Plotly charts, exportable/downloadable CSVs, distinct future model/metadata artifacts, and a reproducible forecast CLI. All four model types are supported; the default synthetic run retains Seasonal Naive because it won validation. Neither the earlier assessed model nor its metrics/predictions are modified.

The future fit includes former holdout observations as known history. It has a later cutoff than the evaluated fit and no associated future accuracy claim. Dates must be strictly after that cutoff. UI horizon switching takes prefixes of one 30-day generation without retraining; changed source/series/dates/values require matching evaluation. The workflow uses exactly the selected history range. Source actuals after its cutoff never enter features or predictions.

Uncertainty decision: intervals are deliberately unavailable. At most three diagnostic origins per lead time and a selection-used validation window do not supply sufficient separate calibration/coverage evidence. Test residuals are not repurposed; no nominal levels, bounds, or coverage claims are invented. Metadata explains this decision. Enabling intervals later requires an explicit chronological calibration and independent coverage protocol.

Verification: 130 tests passed without warnings, including each model's fresh-fit path, preservation of evaluation evidence, stale/cross-series rejection, horizon and calendar checks, recursive isolation, CSV/artifact round trips, zero demand, and UI workflow. Default demo CLI produced 30 dates from 2024-12-31 through 2025-01-29 after a 2024-12-30 refit cutoff, totalling 1,626 predicted synthetic units; no realized-demand or accuracy claim. Separate local bundle: reports/forecasts/phase5-demo (ignored). No inventory calculations were added.

Implement selected store/family 7/14/30-day forecasting and recursive feature updates.

Acceptance: contiguous future dates, no future actual dependence, finite nonnegative outputs, explicit training cutoff, interactive Plotly history/forecast chart, and export-ready tables. Intervals require documented calibration and coverage; omit when unsupported. Test multi-step recursion and invalid horizons.

### Phase 6 — Inventory planning

Status: complete. Added typed/validated user assumptions, matching-forecast or explicit historical-mean scenarios, demand/variability summaries, safety stock, reorder point, EOQ, inventory position, rounded replenishment quantity, days remaining, receipt-aware daily projections, and shortage/excess explanations. The inventory dashboard provides blank manual fields, an explicitly synthetic example mode, an input/calculation audit, stale-result prevention, and a one-scenario CSV export. Inventory date matches the sales cutoff. Business Insights remains Phase 7.

Implementation decisions: maximum lead/receipt delay 365 whole calendar days; one aggregate PO receipt; optional unknown receipt timing; no initial backorders or carried lost-sales backlog; no SKU/pack/minimum/capacity constraints. Historical SD is a disclosed proxy, not a calibrated error distribution or service guarantee. Lead times beyond forecast coverage reject by default and require explicit historical-mean extension. Forecast-window mean remains the EOQ basis; extension only changes lead-time demand/projection. Days remaining uses on-hand only. Only final order quantity is rounded up. An excess flag compares position with ROP+EOQ and can coexist with shortages caused by late orders.

Verification: 163 tests passed without warnings, including hand-calculated results, zero/invalid inputs, rounding boundaries, receipt timing, explicit extension, stale forecasts, export provenance, form error/synthetic states, and evaluation → forecast → inventory integration. A synthetic arithmetic fixture (daily mean 10, SD 2, lead 4, z=1, K=1, H=2.92, on-hand 20, outstanding 5) yields safety stock 4, ROP 44, EOQ 50, position 25, recommendation 50, and on-hand coverage 2 days. These verify arithmetic, not real business outcomes. README documents formulas, source references, policy assumptions, and operational limits.

Implement validated inputs, average demand, safety stock, reorder point, EOQ, inventory position, replenishment quantity, and days remaining.

Acceptance: hand-calculated unit tests plus zero demand, invalid costs/service levels, zero lead time, outstanding orders, and rounding edges. Clearly explain units, assumptions, family-level limitations, stockout/overstock heuristics, and recommended actions.

### Phase 7 — Dashboard integration and business insights

Connect modules with sidebar filters/navigation, consistent minimal theme, metric cards, Plotly charts, readable tables, and plain-language guidance.

Acceptance: complete data-to-recommendation flow; demand increases/decreases with explicit comparison periods and zero-baseline handling; replenishment, shortage, and excess indicators; rule-based procurement recommendations. Forecast/recommendation CSV exports include dates, selections, units, model, provenance, and assumptions. Loading, empty, invalid, and insufficient-history states are actionable. Avoid decorative animations and implementation details in business flows.

### Phase 8 — Testing, documentation, and deployment readiness

Complete regression tests, reproducible setup, example screenshots, methodology, limitations, and minimal Streamlit hosting instructions.

Acceptance: full suite passes in a clean environment; record exact tested dependencies; no datasets, models, secrets, or environments tracked; README reflects actual results and limitations. Document resource limits and deployment configuration. Actual publication requires an explicit instruction.

## Open decisions

- Real Kaggle data have not been inspected. Schema-compatible fixtures and the demo are verified; full-file performance remains unbenchmarked.
- Full-history forecasting is implemented. Intervals remain unsupported with the present calibration evidence; a future extension requires separate calibration origins and independent coverage assessment. Real-data sample adequacy is not established by the demo.
- Phase 6 defines receipt timing and cost units; production-grade SKU mappings, multiple scheduled receipts, pack constraints, and calibrated service coverage remain outside the current family-level scenario scope.
- Dependency ranges are not a lock file; capture exact tested versions before final portfolio release.
