"""Prediction helpers for saved churn models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path) -> Any:
    """Load a serialized model artifact."""
    return joblib.load(model_path)


def predict_churn_probability(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return churn probabilities for the supplied feature matrix."""
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="churn_probability")
