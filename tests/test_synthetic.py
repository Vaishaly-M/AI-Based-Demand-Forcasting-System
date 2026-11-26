"""Verify demo data contracts and reproducibility."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src.demand_planning.data.synthetic import generate_synthetic_sales


def test_reproducible_complete_daily_panel() -> None:
    """Each date/store/family is unique and every series has full coverage."""
    first = generate_synthetic_sales(days=14)
    pd.testing.assert_frame_equal(first, generate_synthetic_sales(days=14))
    assert len(first) == 14 * 2 * 3
    assert first.id.is_unique
    assert not first.duplicated(["date", "store_nbr", "family"]).any()
    assert first.groupby(["store_nbr", "family"]).size().eq(14).all()
    assert first.sales.ge(0).all()
    assert first.onpromotion.ge(0).all()
    assert first.is_synthetic.all()
    assert not first.isna().any().any()
    assert not first.equals(generate_synthetic_sales(days=14, seed=7))


@pytest.mark.parametrize("kwargs", [
    {"days": 0}, {"days": -1}, {"days": 3651}, {"days": 1.5},
    {"days": True}, {"seed": -1}, {"seed": True},
    {"start_date": "not-a-date"}, {"start_date": "2024-02-30"},
])
def test_invalid_parameters(kwargs: dict[str, object]) -> None:
    """Invalid generator inputs fail with an actionable validation error."""
    with pytest.raises(ValueError):
        generate_synthetic_sales(**kwargs)


def test_cli_round_trip_and_overwrite_protection(tmp_path: Path) -> None:
    """CLI writes a usable CSV and cannot silently replace existing data."""
    output = tmp_path / "demo" / "sales.csv"
    command = [sys.executable, "-m", "scripts.generate_synthetic_data",
               "--days", "7", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    frame = pd.read_csv(output)
    assert len(frame) == 42
    assert frame.is_synthetic.all()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert output.read_bytes() == original
    result = subprocess.run(command + ["--overwrite"], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == original
