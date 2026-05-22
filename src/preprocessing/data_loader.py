"""Data loading helpers for customer churn datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_customer_data(path: str | Path) -> pd.DataFrame:
    """Load a customer dataset from CSV or Parquet."""
    data_path = Path(path)
    suffix = data_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".parquet":
        return pd.read_parquet(data_path)

    raise ValueError(f"Unsupported data format: {suffix}")
