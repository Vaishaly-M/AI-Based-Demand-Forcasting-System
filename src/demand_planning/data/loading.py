"""Strict CSV validation with an auditable, non-destructive cleaning report."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import csv

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "store_nbr", "family", "sales"}
OPTIONAL_COLUMNS = {"id", "onpromotion", "is_synthetic"}


class SalesDataError(ValueError):
    """A sales input cannot safely be used for analysis."""


@dataclass
class SalesDataset:
    """Validated observations with source identity and cleaning evidence."""

    frame: pd.DataFrame
    source: str
    provenance: str
    report: pd.DataFrame
    coverage: pd.DataFrame


def validate_sales(frame: pd.DataFrame, source: str = "Provided data") -> SalesDataset:
    """Normalize whitespace/types; reject invalid rows and duplicate keys.

    Missing dates remain absent observations, never imputed zeros. Coverage is
    measured inside each series' observed date span, not presumed store life.
    """
    if frame.empty:
        raise SalesDataError("The dataset contains no sales rows. Supply a labelled historical sales CSV.")
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    if clean.columns.duplicated().any():
        raise SalesDataError("Column names must be unique after removing surrounding whitespace.")
    missing = REQUIRED_COLUMNS - set(clean.columns)
    if missing:
        raise SalesDataError(f"Missing required columns: {', '.join(sorted(missing))}. Use historical train.csv, not unlabelled test.csv.")
    ignored = sorted(set(clean.columns) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
    clean = clean[[column for column in clean if column in REQUIRED_COLUMNS | OPTIONAL_COLUMNS]].copy()
    actions = [("Input rows", str(len(clean))), ("Source", source),
               ("Unused columns excluded", ", ".join(ignored) or "None")]
    renamed = sum(str(a) != b for a, b in zip(frame.columns, [str(c).strip() for c in frame.columns]))
    actions.append(("Headers trimmed", str(renamed)))
    errors: list[str] = []
    for column in ("date", "family"):
        original = clean[column].astype("string")
        trimmed = original.str.strip()
        actions.append((f"{column} values trimmed", str(int(original.ne(trimmed).sum()))))
        clean[column] = trimmed
    dates = pd.to_datetime(clean.date, format="%Y-%m-%d", errors="coerce")
    invalid_dates = ~clean.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False) | dates.isna()
    if invalid_dates.any():
        errors.append(f"date: {int(invalid_dates.sum())} invalid values; use YYYY-MM-DD")
    clean["date"] = dates
    invalid_family = clean.family.isna() | clean.family.eq("")
    if invalid_family.any():
        errors.append(f"family: {int(invalid_family.sum())} missing/empty values")
    for column in ("sales", "store_nbr", "onpromotion"):
        if column not in clean:
            continue
        values = pd.to_numeric(clean[column], errors="coerce").astype(float)
        invalid = ~np.isfinite(values) | values.lt(0)
        if column != "sales":
            invalid |= values.mod(1).ne(0) | values.ge(2**63)
        if column == "store_nbr":
            invalid |= values.eq(0)
        if invalid.any():
            rule = "finite nonnegative units" if column == "sales" else "valid nonnegative integers (stores must be positive)"
            errors.append(f"{column}: {int(invalid.sum())} invalid values; require {rule}")
        else:
            clean[column] = values if column == "sales" else values.astype("int64")
    if "id" in clean:
        ids = clean.id.astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
            errors.append("id: identifiers must be nonempty and unique when supplied")
        clean["id"] = ids
    provenance = "Unverified source (no synthetic flag supplied)"
    if "is_synthetic" in clean:
        flags = clean.is_synthetic.astype("string").str.strip().str.lower()
        if not flags.isin(["true", "false"]).all():
            errors.append("is_synthetic: every value must be true or false when supplied")
        else:
            clean["is_synthetic"] = flags.eq("true")
            provenance = ("Synthetic demonstration data" if clean.is_synthetic.all()
                          else "Mixed synthetic and unverified observations" if clean.is_synthetic.any()
                          else "User-labelled non-synthetic data (not independently verified)")
    if errors:
        raise SalesDataError("Data validation failed. No rows were silently removed. " + "; ".join(errors))
    duplicates = clean.duplicated(["date", "store_nbr", "family"], keep=False)
    if duplicates.any():
        raise SalesDataError(f"Found {int(duplicates.sum())} rows with duplicate date/store/family keys. Resolve duplicates at the source; automatic summing could double-count sales.")
    clean["family"] = clean.family.astype("category")
    clean = clean.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    coverage = clean.groupby(["store_nbr", "family"], observed=True).agg(
        first_date=("date", "min"), last_date=("date", "max"), observed_days=("date", "size")
    ).reset_index()
    coverage["calendar_days"] = (coverage.last_date - coverage.first_date).dt.days + 1
    coverage["missing_days"] = coverage.calendar_days - coverage.observed_days
    actions.extend([
        ("Output rows", str(len(clean))), ("Rows dropped / sales imputed", "0 / 0"),
        ("Types normalized", "ISO dates, numeric units/counts, categorical family"),
        ("Order", "store_nbr, family, date"),
        ("Missing dates inside series spans", str(int(coverage.missing_days.sum()))),
        ("Provenance", provenance),
    ])
    return SalesDataset(clean, source, provenance, pd.DataFrame(actions, columns=["Check / action", "Result"]), coverage)


def load_sales_csv(source: Path | bytes, label: str | None = None) -> SalesDataset:
    """Read a UTF-8 CSV from a local path or uploaded bytes, then validate it.

    Read as strings to preserve identifiers and report malformed numerics rather
    than letting parser inference change their meaning. Raw files are not edited.
    """
    try:
        handle = source.open("rb") if isinstance(source, Path) else BytesIO(source)
        with handle:
            header_line = handle.readline().decode("utf-8-sig")
            header = next(csv.reader([header_line]), [])
            normalized = [column.strip() for column in header]
            if len(normalized) != len(set(normalized)):
                raise SalesDataError("CSV headers must be unique after trimming whitespace.")
            handle.seek(0)
            frame = pd.read_csv(handle, dtype="string", encoding="utf-8-sig")
            if not isinstance(frame.index, pd.RangeIndex):
                raise SalesDataError("CSV rows contain more fields than the header. Correct the comma-separated column counts.")
    except SalesDataError:
        raise
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise SalesDataError("Could not read this CSV. Check that it exists, uses UTF-8, and has a header with consistent comma-separated fields.") from exc
    return validate_sales(frame, label or (source.name if isinstance(source, Path) else "Uploaded CSV"))
