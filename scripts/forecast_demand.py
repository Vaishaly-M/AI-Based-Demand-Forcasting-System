"""Evaluate once, refit the validation winner, and save future demand forecasts."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.data.loading import load_sales_csv, validate_sales
from src.demand_planning.data.synthetic import generate_synthetic_sales
from src.demand_planning.evaluation.artifacts import save_evaluation
from src.demand_planning.evaluation.experiment import run_evaluation
from src.demand_planning.forecasting.artifacts import save_forecast
from src.demand_planning.forecasting.future import forecast_selected_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the complete reproducible selected-model workflow for one series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Omit for reproducible synthetic demo sales")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--family", default="DAIRY")
    parser.add_argument("--horizon", type=int, choices=[7, 14, 30], default=30)
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are protected")
    args = parser.parse_args()
    output = args.output or ROOT / "reports/forecasts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    try:
        if output.exists():
            raise ValueError("Output directory already exists. Choose a new path.")
        dataset = load_sales_csv(args.csv) if args.csv else validate_sales(generate_synthetic_sales(), "Built-in synthetic demo, seed 42")
        selected = dataset.frame.loc[dataset.frame.store_nbr.eq(args.store) & dataset.frame.family.eq(args.family)]
        if selected.empty:
            raise ValueError("No rows match the requested store and family.")
        history = daily_sales(dataset.frame, args.store, args.family, selected.date.min(), selected.date.max())
        evaluation = run_evaluation(history, dataset.source, dataset.provenance, args.store, args.family)
        forecast = forecast_selected_model(history, evaluation, args.horizon, source=dataset.source,
                                           provenance=dataset.provenance, store=args.store, family=args.family)
        save_forecast(forecast, output)
        save_evaluation(evaluation, output / "evaluation")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Forecast could not complete: {exc}\n")
    print(f"Source: {dataset.provenance}")
    print(f"Validation-selected model: {forecast.model.name}")
    print(f"Future refit cutoff: {forecast.model.trained_through.date()}")
    print(f"Forecast dates: {forecast.metadata['forecast_start']} to {forecast.metadata['forecast_end']}")
    print(f"Forecast units: {forecast.forecasts.forecast.sum():.2f}; no future accuracy claim or intervals")
    print(f"Saved future artifacts and separate evaluation evidence to {output}")


if __name__ == "__main__":
    main()
