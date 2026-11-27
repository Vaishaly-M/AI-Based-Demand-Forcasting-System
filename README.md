# AI-Based Demand Forecasting & Inventory Planning System

A Python portfolio project demonstrating retail analytics, machine learning, and supply chain decision support.

**Status: Phase 6 complete.** Sales analytics, model evaluation, future forecasts, and inventory planning scenarios are implemented. Inventory uses explicit user assumptions or labelled synthetic examples, with EOQ/reorder calculations, receipt-aware shortage indicators, excess-stock heuristics, and CSV export. Cross-module Business Insights integration remains planned. Calibrated forecast intervals and service-level guarantees remain unavailable. Demonstrated scores are synthetic experiment results, not real-world performance or business impact claims.

## Windows PowerShell quick start

Requires Python 3.11+. Run from the repository root. Reuse the existing `venv` by skipping the first command if it already exists.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m scripts.generate_synthetic_data
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If Python 3.12 is installed instead, use `py -3.12 -m venv venv`. Open the local URL printed by Streamlit, normally http://localhost:8501. Stop with Ctrl+C. The built-in synthetic demo runs without generating or downloading a CSV; the generator command is optional.

Optional activation:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

If activation is blocked by PowerShell policy, use the explicit interpreter commands above; no policy change is needed.

## Synthetic demonstration data

The generator creates `data/synthetic/sales.csv`: 2,190 rows (365 days × 2 stores × 3 families), starting 2024-01-01, with illustrative weekly/annual seasonality, trend, promotions, noise, and occasional zero sales. Every row has `is_synthetic=True`. These are fabricated demonstration sales, not real business observations; their patterns cannot establish real-world performance.

Existing files are protected. To deliberately regenerate:

```powershell
.\venv\Scripts\python.exe -m scripts.generate_synthetic_data --days 365 --seed 42 --overwrite
```

Additional options: `--start-date YYYY-MM-DD` and `--output data/synthetic/example.csv`. Equal arguments and dependency versions reproduce the CSV. Generated CSVs are ignored by Git. Requirements have bounded version ranges, not an exact lock; record installed versions for experiments.

## Required schema and real data

The intended primary dataset is [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Sign in, review and accept applicable terms, download the files, and extract `train.csv` to `data/raw/train.csv`. Choose **Local Kaggle train.csv** in the sidebar. No Kaggle API or credentials are needed by the application. Do not commit the data or redistribute it without checking its terms. The full competition file has not been downloaded or benchmarked in this workspace.

| Column | Contract |
| --- | --- |
| `date` | Required ISO daily date |
| `store_nbr` | Required positive integer store ID |
| `family` | Required nonempty product-family string |
| `sales` | Required finite nonnegative sales units; fractional values allowed |
| `id` | Optional unique identifier; included in synthetic output |
| `onpromotion` | Optional nonnegative promotional-item count, not a discount |
| `is_synthetic` | Optional provenance flag; always true in generated data |

Grain: one row per date/store/family. The loader validates types, duplicate keys, missing dates, and invalid values; missing dates never silently become zero sales. The competition's unlabelled `test.csv` is not this project's evaluation holdout: chronological holdouts will come from historical labelled sales. Supplementary store/holiday/oil/transaction files are optional later inputs.

## Using Sales Analytics

1. Choose **Synthetic demo**, **Upload CSV**, or **Local Kaggle train.csv** in the sidebar. Upload accepts UTF-8 CSV files, including the generated `data/synthetic/sales.csv`. The built-in demo uses the same generator directly in memory.
2. Choose a store, product family, and analysis date range. Changing store/family resets the range to that series' observed span. Filters persist across navigation.
3. Open **Sales Analytics**. Expand the data-quality report for cleaning actions and per-series coverage. Switch between daily, weekly, and monthly trends; hover over charts for values/sample sizes.
4. Review descriptive weekday/month patterns and family rankings. Run Model Evaluation, then open Demand Forecasting to refit its selected model. Inventory Planning can use that forecast or an explicitly chosen historical-mean scenario. Business Insights remains a placeholder.

**Cleaning policy:** trim header/date/family whitespace, normalize dates and numeric types, exclude unused columns, and sort by store/family/date. Reject the whole input for missing required columns, invalid values, duplicate date/store/family keys, duplicate/nonempty-ID violations, invalid optional promotion/flag values, or malformed CSV structure. No sales rows are silently dropped, duplicate sales summed, or missing demand imputed. Raw files are not modified. Errors explain what must be corrected.

**Missing data:** a full daily calendar is constructed for the selected range; unobserved dates are `NaN`, while recorded zero sales remain zero. The coverage report measures gaps only between each series' first/last observation. It does not infer store opening dates. An all-missing selection displays an empty state.

**KPI definitions:** total sales and average daily demand use observed days only, with coverage shown alongside them. Average sales is a demand proxy, not a stockout-adjusted estimate. Growth compares the final seven calendar days against the preceding seven within the selected window. It is N/A if either window has gaps, fewer than 14 days are selected, or the previous total is zero; a fall from positive demand to zero is -100%.

**Calendar comparisons:** weeks run Monday–Sunday; months use actual calendar lengths. Partial boundary periods and periods with gaps are marked incomplete. Their observed totals are not full-period comparisons. All-missing totals remain blank. Seasonal charts show observed-day means with sample counts, not proof of recurring seasonality; fewer than two years of observations triggers a caution.

**Family rankings:** compare all families in the selected store over the same date window, independently of the selected family. Only complete histories receive a rank; incomplete families remain visible in the coverage table. Rankings use total observed sales, with equal totals tied. Top/bottom five lists may overlap on small datasets. Aggregated family units are not necessarily economically comparable.

**Provenance and scale:** every source is identified. Synthetic flags label demo or mixed observations; absent/false flags do not certify a file as real business data. Validated inputs are cached locally in the running Streamlit process. CSV validation is in memory; use the local source for files exceeding the uploader limit, and allow sufficient RAM for the full Kaggle file. Large-file memory/runtime performance is not yet benchmarked.

Sales are units, not revenue, and may understate demand during stockouts. Inventory levels, purchase orders, lead times, and ordering/holding costs must be provided by users. Product families contain heterogeneous items; family-level inventory planning is illustrative, not an executable SKU purchase order.

## Phase 3: baseline forecasting and features

For the original baseline workflow, open **Demand Forecasting**, choose **Seasonal Naive preview**, then select a **7-, 14-, or 30-day** horizon. The selected end date is the forecast origin; all source observations after it are excluded. An earlier end date produces a historical-origin preview, not an accuracy evaluation. Forecast dates are future relative to the data cutoff, not necessarily today's date. The page shows source provenance, forecast units, a history/forecast chart, and a dated table. No confidence bands or purchasing recommendations are provided. The selected-model workflow is described under Phase 5 below.

The **Seasonal Naive (7-day)** baseline repeats the final observed week on matching weekdays. For example, final-week demand `[1, 2, 3, 4, 5, 6, 7]` yields the same sequence repeated over 14 days and continued for 30 days. Beyond the first week this equals recursion using the prediction from seven days earlier. It needs a complete final seven-day window; an earlier gap is allowed because the baseline does not use it. Missing final-week values block forecasting rather than becoming zero. Negative or infinite history is rejected; outputs are nonnegative. This benchmark does not model promotions, trend, or annual seasonality.

Reusable functions in `src/demand_planning/`:

| Function | Contract |
| --- | --- |
| `data.time_series.validate_daily_series` | Require sorted, unique, timezone-naive midnight dates on a consecutive daily calendar; keep explicit unknown sales as NaN |
| `features.calendar_lags.build_features` | Calendar fields, weekday sine/cosine, lags 1/7/14/28, and prior 7/14/28-day rolling means and population standard deviations (`ddof=0`) |
| `features.calendar_lags.next_day_features` | Create one future feature row from 28 consecutive known history values; append each prediction before constructing another step |
| `forecasting.seasonal_naive.seasonal_naive_forecast` | Forecast 7/14/30 days from supplied history only; return dates, values, method, and forecast origin |
| `evaluation.splits.chronological_split` | Split raw daily observations chronologically before any learned preprocessing or model fitting |

All target-based features at date `t` use sales strictly before `t`. Rolling calculations shift sales by one day and require a full window. The first 28 historical rows contain warm-up NaNs; gaps also propagate NaNs into affected lags/windows. Functions do not silently fill or drop rows. Calendar features contain no inferred holidays or future promotions. Features are generated for one store/family series at a time.

**Important evaluation distinction:** historical feature rows are one-step training examples. Do not build a feature matrix on all validation/test actuals and use it for a fixed-origin multi-step forecast: later lags would leak held-out outcomes. Split raw observations first, fit using training examples only, then build each future row from origin history plus predictions. `next_day_features` supplies that boundary; `predict_recursive` implements it for evaluation and future forecasts. Seasonal Naive does not require the engineered features.

**Split defaults:** final 30 calendar days = test; preceding 30 = validation; everything earlier = training, with a minimum of 56 training days (28 warm-up + at least 28 feature-complete training rows). At least 116 complete daily observations are required. Gaps block split creation until an explicit modelling policy is chosen. Holdout lengths may be increased, but each must remain at least 30 days to support the maximum forecast horizon. The minimum is a structural check, not evidence that the sample is adequate for ML.

For each default 365-day synthetic series, the boundaries are:

| Partition | Inclusive dates | Days |
| --- | --- | --- |
| Train | 2024-01-01 to 2024-10-31 | 305 |
| Validation | 2024-11-01 to 2024-11-30 | 30 |
| Test | 2024-12-01 to 2024-12-30 | 30 |

Phase 3 established these boundaries without scoring either holdout. Phase 4 implements the evaluation below. The interactive baseline preview is exploratory, separate from formal evaluation. Real-data split dates depend on the supplied series; the default policy is computed from its end date, not hardcoded to this demo.

## Phase 4: model training and evaluation

Open **Model Evaluation**, review the dates, then click **Run model evaluation**. At least 116 complete daily observations are required in the selected range. Training is explicit, not triggered on every navigation. Changing the selected series, values, dates, or provenance invalidates the displayed result. The page shows validation scores, earlier rolling-origin diagnostics, the selected model's test scores, and actual-versus-predicted charts.

All four candidates use identical origins and 7/14/30-day horizons. The shorter horizons are prefixes of each 30-day forecast, not independent samples. No hyperparameter search or early stopping is performed in this initial comparison:

| Model | Fixed settings |
| --- | --- |
| Seasonal Naive | Repeat the final seven observed days |
| Linear Regression | StandardScaler + ordinary least squares; scaler fitted only on training feature rows |
| Random Forest | 100 trees, maximum depth 10, minimum leaf size 2, seed 42, one worker |
| XGBoost | 120 trees, depth 3, learning rate 0.05, full row/column sampling, squared-error objective, histogram trees, seed 42, one worker |

Models train on calendar/lag/rolling features after removing exactly the 28 warm-up rows; no missing data are imputed. Each recursive forecast builds the next day's features from history and prior predictions. Negative predictions are clipped **before** feedback and scoring; nonfinite outputs abort evaluation. Future actuals never populate recursive lags. Trees require no scaling. Linear Regression's scaler is fitted independently inside each training window.

Evaluation sequence:

1. Reserve the final 30 days for test and preceding 30 for dedicated validation.
2. Run up to three expanding-window diagnostic folds **within the training partition**. Each holdout is 30 days; origins are 30 days apart and require at least 56 earlier training days. Short datasets may have fewer folds or none. These results are not the selection score.
3. Fit all candidates on training only, forecast the dedicated validation period, and choose the lowest **30-day validation WAPE**. If all validation demand is zero, use MAE. Ties resolve by MAE, RMSE, then the fixed order Seasonal Naive, Linear Regression, Random Forest, XGBoost. Test outcomes never influence this choice.
4. Refit only the winner on train plus validation and forecast the 30-day test once per run. Report its 7/14/30-day metrics. There is no test leaderboard and no fit using test targets.

MAE and RMSE are in sales units. WAPE is `100 * sum(abs(error)) / sum(actual)` and is **N/A for an all-zero actual window**, even for a perfect forecast. MAE/RMSE remain available. Empty, mismatched, nonfinite, or negative-actual metric inputs fail rather than being silently excluded. The minimum history is a structural requirement, not statistical evidence of sufficient training data.

Freeze dates/settings before an experiment. Reproducible reruns are possible, but manually inspecting test results and changing dates or settings would invalidate an untouched-test claim. The interactive application cannot prevent that human workflow. The demo reruns used unchanged model settings; no test-based tuning was performed. No prediction intervals or significance claims are provided.

### Reproduce and save locally

```powershell
.\venv\Scripts\python.exe -m scripts.evaluate_models
# Optional real input; use a new output directory for each experiment:
.\venv\Scripts\python.exe -m scripts.evaluate_models --csv data/raw/train.csv --store 1 --family "DAIRY" --output reports/evaluation/my-run
```

The default CLI uses the explicitly synthetic demo, Store 1 / DAIRY, and creates a unique UTC-labelled folder beneath `reports/evaluation/`. The UI's **Save local experiment artifacts** button does the same. Existing run directories cannot be overwritten. Each bundle includes:

- `selected_model.joblib`: winner fitted through validation end, metadata, and the final 28 **pre-test** observations. This is not a full-history production forecast model.
- `metadata.json`: data fingerprint, source/provenance, store/family, inclusive split dates, CV origins, feature definitions, policies, estimator parameters, seed, training cutoff, and dependency/Python versions. Nonfinite library parameter sentinels such as XGBoost's `missing=NaN` are represented as strings for valid JSON.
- `validation_metrics.csv`, `rolling_cv_metrics.csv`, `selected_test_metrics.csv`, and `predictions.csv`: actual run evidence, including dates, origins, model, stage, and predictions. An empty CV file means no extra fold fit the minimum history.

Artifacts are local and ignored by Git. Joblib uses pickle: load only trusted files you created; there is no model-upload loader. Training or persistence failure is reported, and no successful model comparison is fabricated. A failed write can leave an incomplete directory; use a new directory on retry. Forecast CSV download is implemented in Phase 5; a single inventory-scenario CSV is available in Phase 6.

### Observed synthetic demonstration results

For seed 42, Store 1 / DAIRY, 365 days starting 2024-01-01, with the split dates above and unchanged fixed settings:

| Model | 30-day validation MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| Seasonal Naive | 8.0333 | 11.6032 | 16.08% |
| Linear Regression | 10.3816 | 13.4833 | 20.78% |
| Random Forest | 9.2499 | 11.9684 | 18.51% |
| XGBoost | 9.6258 | 13.6328 | 19.26% |

Seasonal Naive was selected. Its **30-day reserved-test** MAE was **14.3667**, RMSE **19.4414**, and WAPE **29.91%**. The baseline won this particular validation comparison; ML improvement is not assumed. Three earlier diagnostic folds were available. The final verified local bundle is `reports/evaluation/phase4-demo-final/` (not committed); regenerate with the CLI. Synthetic scores are not evidence of performance on Kaggle or a real retailer.

## Phase 5: future forecasts from the selected model

1. Select a source, store, family, and date range. At least 116 complete days are required for model evaluation. **All selected history** means that exact date range, not earlier or later rows outside it.
2. On **Model Evaluation**, run the fixed comparison once for that selection.
3. Open **Demand Forecasting** and use **Selected model** (the default). Click **Refit selected model and forecast**. This creates a fresh estimator with the previously fixed settings and the validation-selected model name.
4. Select 7, 14, or 30 days; inspect the Plotly chart and table. Download the forecast CSV or save the future model locally. The UI generates 30 days once and takes prefixes when switching horizons; it does not rerun evaluation or refit for those switches.

The future refit uses every observation through the selected cutoff, including the **former** validation/test windows, because those dates are now known history for forecasting subsequent dates. It does not change model selection, the earlier fitted model, test metrics, or evaluation predictions. Those test scores describe the earlier pre-test fit; they are **not accuracy estimates for the new full-history fit or its unobserved future**. No new scores are calculated during refitting. The selected model may be Seasonal Naive if it won validation; ML is not forced to replace a better baseline.

Selection reuse requires the same source/provenance, store/family, dates, values, and supported experiment version. Changed or appended data need an explicit matching evaluation; the current version does not automatically carry an old selection into a new dataset. In the UI, run evaluation and refit in the same session. Arbitrary saved Joblib files cannot be uploaded or executed. The pure Python service accepts a trusted in-memory `EvaluationResult`.

Future dates are contiguous daily dates strictly after the cutoff. Recorded zeros remain valid, missing selected-history values block the workflow, and outputs are finite/nonnegative. ML models update lag/rolling inputs using clipped prior predictions, never future actuals. The weekly baseline repeats its final observed week. No actual rows after the cutoff are passed to forecasting, even when the source contains them. The chart shows up to 90 historical days and the chosen horizon; dates do not jump to the current real-world date.

### Uncertainty decision

**No interval is published in Phase 5.** The current evaluation produces at most three earlier origins for each lead time. It has no separate interval-calibration sample or independent interval-coverage assessment; the dedicated validation window has already been used for selection. These data do not support defensible horizon-specific empirical intervals. Test residuals are not reused for calibration. CSV/metadata say `uncertainty_status=unavailable`, with no invented bounds or nominal coverage; metadata records `interval_coverage=null`. Future calibration work must define additional chronological calibration origins, preserve an independent coverage assessment, and document dependence/distribution-shift assumptions before enabling bands.

### Reproduce future forecasts

```powershell
# Default synthetic Store 1 / DAIRY, 30 days; evaluation + separate future refit:
.\venv\Scripts\python.exe -m scripts.forecast_demand --horizon 30
# Optional real data and another supported horizon:
.\venv\Scripts\python.exe -m scripts.forecast_demand --csv data/raw/train.csv --store 1 --family "DAIRY" --horizon 14 --output reports/forecasts/my-run
```

The CLI performs evaluation once, then refits without changing settings based on test performance. Default output is a new UTC-labelled directory beneath `reports/forecasts/`; existing directories are protected. Each run saves:

- `forecast.csv`: future date, forecast units, cutoff, model, store/family, source/provenance, units, lead day, horizon, history start, fit scope, selection basis, uncertainty status, and evaluation fingerprint.
- `forecast_metadata.json`: earlier evaluation fit cutoff **and** new full-history cutoff, split/feature/parameter/version provenance, forecast dates, and explicit uncertainty/accuracy limitations.
- `future_model.joblib`: a distinct future model, metadata, and final 28 historical observations. Evaluation artifacts are never overwritten.
- `evaluation/`: the CLI's separate Phase 4 evidence and pre-test model. The UI instead retains evaluation in session and lets you save it separately from Model Evaluation.

The UI's **Download forecast CSV** exports the selected horizon without retraining. **Save future model and forecast locally** saves a future bundle for that horizon. The optional unselected baseline preview can also download a CSV, clearly labelled `Unselected weekly baseline preview`; it has no evaluation selection claim. All generated artifacts remain ignored by Git.

Observed default demo run: validation selected Seasonal Naive, the future refit cutoff was **2024-12-30**, and its 30 forecast dates were **2024-12-31 through 2025-01-29**, totalling **1,626 synthetic forecast units**. That total is a model output, not realized demand or an accuracy claim. The earlier assessed fit still ends 2024-11-30. The verified local bundle is `reports/forecasts/phase5-demo/` (not committed).

## Phase 6: inventory planning scenarios

Open **Inventory Planning** for the selected store/family. The scenario date is the selected sales-history cutoff, **not necessarily today**: inventory and open orders must represent that same snapshot. Family-level units/costs are illustrative aggregates, not ready-to-send SKU purchase orders.

1. Choose **Forecast-based planning** after generating a matching selected-model future forecast. Select a 7/14/30-day demand window (default 30); this is independent of the forecast chart's horizon. Stale or different-source forecasts are rejected. Alternatively choose **Historical mean scenario**, which requires no model and is explicitly labelled as an assumption.
2. Enter current inventory, supplier lead time, cycle service target, ordering cost, annual holding cost, outstanding purchase-order units, and a currency label. Stock, lead-time, and cost fields start blank. The 95% service default is an editable assumption, not measured performance. Choose **Synthetic example** only for clearly labelled prefilled demonstration inputs; edits in that mode remain labelled synthetic.
3. Optionally enter the single aggregate outstanding order's expected receipt day. Leave it blank if unknown. For lead times longer than the forecast window, explicitly enable the historical-mean extension or choose a longer forecast; demand is never silently truncated.
4. Click **Calculate inventory scenario**. Review KPIs, shortage/excess flags, the submitted-input audit, and daily stock projection. Download a one-row scenario CSV containing inputs, calculated outputs, policies, recommendations, and both sales/inventory provenance.

Inputs are finite nonnegative quantities/costs; annual holding cost must be strictly positive. Lead time and optional receipt day are whole calendar days from 0 to 365. Cycle service targets are limited to 50%–99.9%. Holding cost is **currency per unit per year**, not a percentage; ordering cost is currency per order. Currency is a three-letter uppercase label with no conversion or exchange-rate lookup. No existing backorders are assumed. At least two complete selected historical days are needed for variability; gaps are rejected rather than filled.

### Calculation definitions

Let `d` be the mean of the chosen forecast window, or the selected historical mean in historical mode; `L` lead time in days; `sigma` the **sample standard deviation of selected historical daily sales** (`ddof=1`); `z` the standard-normal quantile of the cycle service target; `K` ordering cost; and `H` annual holding cost per unit.

| Output | Implemented rule |
| --- | --- |
| Annualized demand `D` | `365 * d`; extrapolation of the chosen window, not a seasonal annual forecast |
| Lead-time demand | Sum the next `L` forecast days; historical mode uses `L * historical_mean` |
| Safety stock | `z * sigma * sqrt(L)` |
| Reorder point | Expected lead-time demand + safety stock |
| EOQ | `sqrt(2 * D * K / H)` |
| Inventory position | Current on-hand + all outstanding units; zero existing backorders |
| Suggested replenishment | If position <= reorder point: `ceil(max(EOQ, reorder_point - position, 0))`; otherwise 0 |
| Days of stock remaining | Current on-hand / `d`; N/A for zero mean demand; excludes outstanding receipts |
| Potential excess stock | Position strictly exceeds reorder point + one EOQ; a heuristic threshold |

The classical EOQ formula and its steady-demand, full-lot-receipt assumptions are described in [MIT's inventory lectures](https://www.ocw.mit.edu/courses/2-854-introduction-to-manufacturing-systems-fall-2016/6c8fec8f99bcf7059b73b82e96d43901_MIT2_854F16_Inventory.pdf). The fixed-lead-time safety-stock approximation scales demand variability with the square root of lead time; see the [APICS safety-stock article hosted by MIT](https://web.mit.edu/course/2/2.810/www/files/readings/King_SafetyStock.pdf). The combined replenishment rule and excess threshold above are this project's explicitly documented scenario policies, not claims of globally optimal procurement.

**Safety-stock limitation:** raw historical sales SD is a provisional variability proxy, not calibrated lead-time forecast error. Independent daily demand, fixed supplier lead time, and a normal approximation are assumed. Seasonality, autocorrelation, intermittent demand, stockout-censored sales, and variable lead time can invalidate those assumptions. Cycle service is distinct from fill rate; the requested target is not guaranteed and does not establish forecast interval coverage. Two historical days satisfy only a mechanical minimum, not statistical adequacy.

If `L` exceeds the chosen forecast window, the default is to reject the calculation. Explicit extension uses the selected historical mean for the extra days and records the count. EOQ annualization continues to use the chosen forecast-window mean; the extension affects lead-time demand/projection only. All intermediate quantities stay unrounded; only the final suggested replenishment is rounded up. No pack sizes, minimum orders, capacity, discounts, backorder costs, or SKU cost differences are modelled.

### Receipts, shortage risk, and edge cases

Outstanding purchase orders are one aggregate receipt. **Day 0** arrives before the first future day's demand. Any later receipt arrives **at the end of its specified day**, after that day's demand. A normal new replenishment is assumed to arrive at the end of the supplier lead time; the displayed projection assesses the waiting period **without adding the newly recommended order**.

The projection consumes demand from available stock, counts any deficit as unmet/lost sales, then credits scheduled existing receipts. Physical stock is never negative; unmet demand is not carried forward as a backlog. A shortage flag means projected unmet demand on at least one lead-time day, not a probability. Tiny subtraction noise at equal fractional balances is ignored using relative tolerance `1e-12` (no absolute unit tolerance).

Orders with unknown receipt timing count in inventory position but are excluded from the shortage projection. The result requests timing confirmation and labels any shortage as conditional. Late orders can create **both** shortage risk and excess inventory position. In that case the rule may suggest no new quantity while recommending expediting or reviewing existing orders; it does not assume that more normal-lead-time orders can prevent an earlier shortage.

- Zero lead time: zero lead-time demand/buffer and no waiting-day projection; receipt is assumed immediate.
- Zero mean demand: annualized demand and EOQ are zero, and days of stock is N/A. A nonzero historical SD can still create a provisional safety-stock gap; the UI explicitly asks the user to review that assumption. Completely zero history gives zero buffer and no replenishment quantity.
- Zero ordering cost: EOQ is zero in the continuous formula (no positive optimal batch); only the reorder-point gap can suggest quantity. Holding cost of zero is rejected.
- Editing a form requires resubmission; displayed results identify the submitted assumptions. Changed data, dates, family, demand basis, or forecast window clear the old result. Invalid submissions clear the prior recommendation.

### Hand-worked verification example

This **synthetic arithmetic fixture** is separate from the dashboard's convenient prefilled example. Historical demand `[8, 10, 12]` has mean 10 and sample SD 2. Assume on-hand 20, outstanding orders 5 with unknown timing, lead time 4, service target `Phi(1)` (about 84.13%, so `z=1`), ordering cost 1, and annual holding cost 2.92 per unit.

| Result | Expected and verified |
| --- | ---: |
| Annualized demand | 3,650 |
| Lead-time demand | 40 |
| Safety stock | 4 |
| Reorder point | 44 |
| EOQ | 50 |
| Inventory position | 25 |
| Suggested quantity | 50 |
| Days of on-hand stock | 2 |
| First projected shortage day | 3 |
| Projected unmet demand over lead time | 20 |

The five outstanding units are not credited in the projection until receipt timing is supplied. These values verify formula behavior; they are not observed inventory, realized shortages, savings, or purchasing instructions.

The reusable API is `src.demand_planning.inventory.planning`: `InventoryInputs`, `plan_inventory`, and `InventoryPlan.export_frame()`. It accepts validated history and, optionally, a matching `FutureForecast`; it neither trains models nor reads inventory from a sales CSV.

## Structure

```text
streamlit_app.py              Shared shell and sidebar navigation
app_pages/                   Analytics, model evaluation, future forecasts, remaining placeholders
src/demand_planning/
  data/                      Synthetic generation, CSV validation, cleaning report
  analytics/                 Coverage-aware KPIs, trends, profiles, rankings
  features/                  Calendar, lag, rolling, and next-day feature builders
  forecasting/               Models, recursive forecasts, selected-model future refits and artifacts
  evaluation/                Splits, metrics, experiments, and trusted local artifacts
  inventory/                 Validated inputs, inventory rules, receipt projections, scenario export
  insights/                  Reserved for business recommendation rules
scripts/                     Data generator, evaluate_models, and forecast_demand CLI utilities
tests/                       Generator and multipage smoke tests
data/{raw,processed,synthetic}/ Local data, ignored except directory markers
models/                      Ignored trained artifacts
reports/                     Ignored generated reports and exports
.streamlit/config.toml       Consistent minimal dashboard theme
AGENTS.md                    Project-wide development instructions
PROJECT_PLAN.md              Architecture, phases, acceptance criteria
requirements.txt             Runtime and test dependencies
pytest.ini                   Test discovery configuration
```

## Verification

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m pip check
```

Tests check data ingestion, analytics, feature leakage, model evaluation, future forecasts, inventory arithmetic, and all six navigation destinations. They cover training isolation, recursive predictions, artifact round trips, input/provenance validation, receipt timing, zero-demand/cost cases, and stale-result prevention. These checks do not constitute visual browser testing or validate unimplemented functionality.

Phase 6 verification: **163 tests passed without warnings** on Windows/Python 3.12.6. New checks include hand-calculated safety stock/ROP/EOQ, fractional-unit rounding, zero demand/lead time/ordering cost, invalid service/cost inputs, immediate/end-of-day/unknown/late receipts, simultaneous shortage/excess, explicit historical extensions, stale forecasts, CSV provenance, manual/synthetic input UX, and the full evaluation → future forecast → inventory scenario flow. The normal approximation and user inventory assumptions are not validated against real operations.

Phase 5 verification: **130 tests passed without warnings** on Windows/Python 3.12.6. New checks cover future-fit support for all four models, unchanged evaluation estimators/metrics/predictions, date and horizon boundaries, stale/cross-series selection rejection, zero-demand forecasts, forecast prefixes without refitting, isolation from source actuals beyond cutoff, CSV fields, trusted future-artifact round trips, and the evaluate-to-refit UI flow. The default CLI completed and saved separate future/evaluation bundles. Real Kaggle data, future accuracy, interval coverage, and manual browser appearance remain unverified.

Phase 4 verification: **106 tests passed without warnings** on Windows/Python 3.12.6. `pip check` found no broken requirements. The CLI completed the full default synthetic comparison, saved the final bundle, and reproduced unchanged scores after the metadata serialization fix. No real Kaggle file or manual browser appearance has been verified.

Phase 3 verification: **88 tests passed without warnings** on Windows/Python 3.12.6. Unit tests verify that altering validation/test actuals cannot change training features or forecasts issued at the corresponding origin. The synthetic split dates above were computed from the generator. No model accuracy, test-set scores, or business outcomes were calculated.

Phase 2 verification: **55 tests passed** on Windows/Python 3.12.6. Actual default-demo inspection found 2,190 rows across six store/family series, 35 zero-sales rows, and no internal missing dates. Dates span 2024-01-01 through 2024-12-30 (365 days in a leap year); December is correctly marked incomplete. These are synthetic data checks, not business findings or forecasting results. Real Kaggle data and visual browser appearance remain unverified.

Phase 1 verification: **17 tests passed** on Windows/Python 3.12.6; `pip check` found no broken requirements; Python compilation checks passed. The default generator produced 2,190 rows. Git exclusions were checked for the environment, raw/generated data, model artifacts, and secrets. Tested direct dependencies: Streamlit 1.67.0, Pandas 2.3.3, NumPy 2.5.5, scikit-learn 1.9.1, XGBoost 3.4.1, Plotly 6.9.0, Joblib 1.6.0, and Pytest 9.2.0. Other Python versions have not been tested in this workspace.

## Roadmap and stack

Implemented: sales analytics, causal features, model evaluation, future forecasts, inventory/replenishment scenarios, receipt-aware risk indicators, charts, CSV downloads, and local audit artifacts. Planned: cross-module Business Insights integration and final testing/documentation/deployment readiness. Calibrated forecast uncertainty and guaranteed inventory service levels remain unavailable.

Stack: Python, Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Plotly, Joblib, and Pytest. No database or separate API is required. Navigation follows the [Streamlit multipage API](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation). Deployment preparation belongs to Phase 8; no deployment has been performed.

Next: **Phase 7 — dashboard integration and business insights**, after approval. See [PROJECT_PLAN.md](PROJECT_PLAN.md).
