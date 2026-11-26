# AI-Based Demand Forecasting & Inventory Planning System

A Python portfolio project demonstrating retail analytics, machine learning, and supply chain decision support.

**Status: Phase 1 foundation only.** A multipage dashboard with clearly labelled placeholders and a reproducible synthetic sales generator are implemented. Analytics, trained models, inventory calculations, charts, and exports are planned. No accuracy or business impact is claimed.

## Windows PowerShell quick start

Requires Python 3.11+. Run from the repository root. Reuse the existing `venv` by skipping the first command if it already exists.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m scripts.generate_synthetic_data
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If Python 3.12 is installed instead, use `py -3.12 -m venv venv`. Open the local URL printed by Streamlit, normally http://localhost:8501. Stop with Ctrl+C. The app runs without generating or downloading data; Phase 1 does not load CSV files yet.

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

The intended primary dataset is [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Sign in, review and accept applicable terms, download the files, and extract `train.csv` to `data/raw/train.csv`. No Kaggle API or credentials are needed by this scaffold. Do not commit the data or redistribute it without checking its terms. Loading is planned for Phase 2.

| Column | Contract |
| --- | --- |
| `date` | Required ISO daily date |
| `store_nbr` | Required positive integer store ID |
| `family` | Required nonempty product-family string |
| `sales` | Required finite nonnegative sales units; fractional values allowed |
| `id` | Optional unique identifier; included in synthetic output |
| `onpromotion` | Optional nonnegative promotional-item count, not a discount |
| `is_synthetic` | Optional provenance flag; always true in generated data |

Grain: one row per date/store/family. Phase 2 will validate types, duplicates, missing dates, and invalid values; missing dates must not silently become zero sales. The competition's unlabelled `test.csv` is not this project's evaluation holdout: chronological holdouts will come from historical labelled sales. Supplementary store/holiday/oil/transaction files are optional later inputs.

Sales are units, not revenue, and may understate demand during stockouts. Inventory levels, purchase orders, lead times, and ordering/holding costs must be provided by users. Product families contain heterogeneous items; family-level inventory planning is illustrative, not an executable SKU purchase order.

## Structure

```text
streamlit_app.py              Shared shell and sidebar navigation
app_pages/                   Overview and five module placeholders
src/demand_planning/
  data/                      Synthetic generation; future CSV validation
  features/                  Reserved for calendar, lag, rolling features
  forecasting/               Reserved for baseline and ML models
  evaluation/                Reserved for temporal evaluation and metrics
  inventory/                 Reserved for inventory calculations
  insights/                  Reserved for business recommendation rules
scripts/                     Command-line utilities
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

Tests check deterministic generation, complete unique daily panels, nonnegative sales, synthetic labels, invalid inputs, and all six navigation destinations using Streamlit AppTest. Feature, metric, and inventory unit tests will accompany their implementations. These checks do not constitute visual browser testing or validate unimplemented functionality.

Phase 1 verification: **17 tests passed** on Windows/Python 3.12.6; `pip check` found no broken requirements; Python compilation checks passed. The default generator produced 2,190 rows. Git exclusions were checked for the environment, raw/generated data, model artifacts, and secrets. Tested direct dependencies: Streamlit 1.67.0, Pandas 2.3.3, NumPy 2.5.5, scikit-learn 1.9.1, XGBoost 3.4.1, Plotly 6.9.0, Joblib 1.6.0, and Pytest 9.2.0. Other Python versions have not been tested in this workspace.

## Roadmap and stack

Planned: sales KPIs and filtering; seasonal naive, Linear Regression, Random Forest, and XGBoost; 7/14/30-day forecasts; chronological validation and untouched test evaluation using MAE/RMSE/WAPE; user-supplied inventory scenarios; explainable recommendations; Plotly charts and CSV exports. Uncertainty will be shown only with a supported evaluation methodology.

Stack: Python, Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Plotly, Joblib, and Pytest. No database or separate API is required. Navigation follows the [Streamlit multipage API](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation). Deployment preparation belongs to Phase 8; no deployment has been performed.

Next: **Phase 2 — dataset preparation and exploratory analysis**, after approval. See [PROJECT_PLAN.md](PROJECT_PLAN.md).
