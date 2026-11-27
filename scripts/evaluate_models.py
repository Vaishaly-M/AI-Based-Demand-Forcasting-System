"""Reproduce an evaluation and save a new trusted local artifact bundle."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.demand_planning.analytics.sales import daily_sales
from src.demand_planning.data.loading import load_sales_csv, validate_sales
from src.demand_planning.data.synthetic import generate_synthetic_sales
from src.demand_planning.evaluation.artifacts import save_evaluation
from src.demand_planning.evaluation.experiment import run_evaluation

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Evaluate one selected series using fixed splits/settings, without tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Omit to use explicitly synthetic demo sales")
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--family", default="DAIRY")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    output = args.output or ROOT / "reports/evaluation" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    try:
        if output.exists():
            raise ValueError("Output directory already exists. Choose a new path to preserve previous experiments.")
        dataset = load_sales_csv(args.csv) if args.csv else validate_sales(generate_synthetic_sales(), "Built-in synthetic demo, seed 42")
        selected = dataset.frame.loc[dataset.frame.store_nbr.eq(args.store) & dataset.frame.family.eq(args.family)]
        if selected.empty:
            raise ValueError("No rows match the requested store and family.")
        history = daily_sales(dataset.frame, args.store, args.family, selected.date.min(), selected.date.max())
        result = run_evaluation(history, dataset.source, dataset.provenance, args.store, args.family)
        save_evaluation(result, output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Evaluation could not complete: {exc}\n")
    print(f"Source: {dataset.provenance}")
    print(f"Selected by validation: {result.selected_model.name}")
    print(result.validation_metrics.loc[result.validation_metrics.horizon.eq(30)].to_string(index=False))
    print("Selected model test results (no test-based selection):")
    print(result.test_metrics.to_string(index=False))
    print(f"Artifacts saved to {output}")


if __name__ == "__main__":
    main()
