"""Run from the repository root: python -m scripts.generate_synthetic_data."""

import argparse
from pathlib import Path

from src.demand_planning.data.synthetic import generate_synthetic_sales

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write a deterministic synthetic CSV, protecting existing files by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--output", type=Path, default=ROOT / "data/synthetic/sales.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        frame = generate_synthetic_sales(args.days, args.seed, args.start_date)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.2f")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Could not generate synthetic sales: {exc}\n")
    print(f"Created {len(frame):,} explicitly synthetic sales rows at {args.output}")


if __name__ == "__main__":
    main()
