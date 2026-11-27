"""Sales input contract, failure policy, and source provenance tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.demand_planning.data.loading import SalesDataError, load_sales_csv, validate_sales


@pytest.fixture
def sales_frame() -> pd.DataFrame:
    """A small valid but gapped series with genuine zero and fractional sales."""
    return pd.DataFrame({"date": ["2024-01-01", "2024-01-03"], "store_nbr": [1, 1],
                         "family": ["DAIRY", "DAIRY"], "sales": [0, 2.5]})


def test_cleaning_keeps_values_and_exposes_gaps(sales_frame: pd.DataFrame) -> None:
    original = sales_frame.copy(deep=True)
    sales_frame["family"] = " DAIRY "
    sales_frame = sales_frame.rename(columns={"sales": " sales "})
    result = validate_sales(sales_frame, "fixture")
    assert result.frame.sales.tolist() == [0, 2.5]
    assert result.frame.family.tolist() == ["DAIRY", "DAIRY"]
    assert result.coverage.missing_days.tolist() == [1]
    assert result.source == "fixture"
    assert "Unverified" in result.provenance
    assert " sales " in sales_frame  # caller-owned input not changed
    assert original.sales.tolist() == [0, 2.5]


@pytest.mark.parametrize("column,value", [
    ("sales", -1), ("sales", np.inf), ("sales", np.nan), ("sales", "invalid"),
    ("store_nbr", 0), ("store_nbr", 1.5), ("store_nbr", -1),
    ("family", " "), ("family", None), ("date", "2024-02-30"),
    ("date", "2024-01-01T12:00:00"), ("date", "01/01/2024"),
    ("onpromotion", -1), ("onpromotion", 0.5), ("is_synthetic", "maybe"),
])
def test_bad_values_reject_entire_input(sales_frame: pd.DataFrame, column: str, value: object) -> None:
    sales_frame[column] = pd.Series([value, value], dtype=object)
    with pytest.raises(SalesDataError, match=column):
        validate_sales(sales_frame)


def test_duplicate_keys_after_cleaning_rejected(sales_frame: pd.DataFrame) -> None:
    sales_frame["date"] = "2024-01-01"
    sales_frame["family"] = ["DAIRY", " DAIRY "]
    with pytest.raises(SalesDataError, match="duplicate"):
        validate_sales(sales_frame)


def test_optional_id_must_be_unique(sales_frame: pd.DataFrame) -> None:
    sales_frame["id"] = ["1", "1"]
    with pytest.raises(SalesDataError, match="id"):
        validate_sales(sales_frame)


@pytest.mark.parametrize("flags,expected", [
    ([True, True], "Synthetic demonstration"),
    ([True, False], "Mixed synthetic"),
    ([False, False], "not independently verified"),
])
def test_provenance_is_not_inferred_from_filename(sales_frame: pd.DataFrame, flags: list[bool], expected: str) -> None:
    sales_frame["is_synthetic"] = flags
    assert expected in validate_sales(sales_frame, "real_business.csv").provenance


@pytest.mark.parametrize("payload", [
    b"", b"date,store_nbr,family,sales\n", b"date,store_nbr,family\n2024-01-01,1,A\n",
    b"date,store_nbr,family,sales,sales\n2024-01-01,1,A,1,2\n",
    b"date,store_nbr,family,sales\nextra,2024-01-01,1,A,1\n",
    b"\xff\xff", b'date,store_nbr,family,sales\n"unterminated',
])
def test_invalid_csv_has_actionable_error(payload: bytes) -> None:
    with pytest.raises(SalesDataError):
        load_sales_csv(payload)


def test_local_and_uploaded_csv_match(tmp_path: Path, sales_frame: pd.DataFrame) -> None:
    path = tmp_path / "train.csv"
    sales_frame.to_csv(path, index=False)
    local = load_sales_csv(path)
    uploaded = load_sales_csv(path.read_bytes(), "upload")
    pd.testing.assert_frame_equal(local.frame, uploaded.frame)
    assert local.source == "train.csv"
    assert uploaded.source == "upload"
    with pytest.raises(SalesDataError, match="Could not read"):
        load_sales_csv(tmp_path / "missing.csv")
