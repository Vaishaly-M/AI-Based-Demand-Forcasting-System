"""Shared source selection and filters, separate from data validation logic."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.demand_planning.data.loading import SalesDataError, SalesDataset, load_sales_csv, validate_sales
from src.demand_planning.data.synthetic import generate_synthetic_sales

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(max_entries=2, show_spinner=False)
def demo_dataset() -> SalesDataset:
    """Build the reproducible demonstration without requiring a local CSV."""
    return validate_sales(generate_synthetic_sales(), "Built-in demo: seed 42, 365 days")


@st.cache_data(max_entries=2, show_spinner=False)
def uploaded_dataset(content: bytes, name: str) -> SalesDataset:
    """Cache validation for unchanged uploads."""
    return load_sales_csv(content, f"Uploaded: {name}")


@st.cache_data(max_entries=2, show_spinner=False)
def local_dataset(path: str, modified_ns: int, size: int) -> SalesDataset:
    """Invalidate cached local data when file timestamp or size changes."""
    return load_sales_csv(Path(path))


def render_data_controls() -> None:
    """Populate shared session context, clearing it on missing/invalid inputs."""
    st.session_state["sales_dataset"] = None
    st.session_state["sales_selection"] = None
    source = st.sidebar.selectbox("Sales data", ["Synthetic demo", "Upload CSV", "Local Kaggle train.csv"], key="data_source")
    try:
        with st.spinner("Loading and validating sales data..."):
            if source == "Synthetic demo":
                dataset = demo_dataset()
            elif source == "Upload CSV":
                upload = st.sidebar.file_uploader("Historical sales CSV (UTF-8)", type=["csv"], key="sales_upload")
                if upload is None:
                    st.sidebar.info("Upload a CSV containing date, store_nbr, family, and sales.")
                    return
                dataset = uploaded_dataset(upload.getvalue(), upload.name)
            else:
                path = ROOT / "data/raw/train.csv"
                if not path.is_file():
                    st.sidebar.info("Place Kaggle train.csv in data/raw/, then rerun. See README for download instructions.")
                    return
                metadata = path.stat()
                dataset = local_dataset(str(path), metadata.st_mtime_ns, metadata.st_size)
    except (SalesDataError, OSError) as exc:
        st.sidebar.error(str(exc) if isinstance(exc, SalesDataError) else "The local file is unavailable. Check permissions and retry.")
        return
    st.session_state["sales_dataset"] = dataset
    st.sidebar.caption(dataset.provenance)
    stores = sorted(dataset.frame.store_nbr.unique().tolist())
    if st.session_state.get("selected_store") not in stores:
        st.session_state["selected_store"] = stores[0]
    store = st.sidebar.selectbox("Store", stores, key="selected_store")
    store_frame = dataset.frame.loc[dataset.frame.store_nbr.eq(store)]
    families = sorted(store_frame.family.unique().tolist())
    if st.session_state.get("selected_family") not in families:
        st.session_state["selected_family"] = families[0]
    family = st.sidebar.selectbox("Product family", families, key="selected_family")
    selected = store_frame.loc[store_frame.family.eq(family)]
    lower, upper = selected.date.min().date(), selected.date.max().date()
    selection_id = (source, dataset.source, int(store), family, lower, upper)
    if st.session_state.get("date_selection_id") != selection_id:
        st.session_state["sales_dates"] = (lower, upper)
        st.session_state["date_selection_id"] = selection_id
    dates = st.sidebar.date_input("Analysis dates", min_value=lower, max_value=upper, key="sales_dates")
    if len(dates) != 2:
        st.sidebar.info("Select both a start and end date.")
        return
    st.session_state["sales_selection"] = (int(store), family, pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
    st.sidebar.caption("Dates select analysis history, forecast cutoff, and the inventory scenario date. Changed data or filters require matching forecasts and resubmitted inventory assumptions.")
