"""Reusable charts for churn analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def churn_distribution(data: pd.DataFrame, target_column: str) -> Figure:
    """Build a churn class distribution chart."""
    counts = data[target_column].value_counts().rename_axis(target_column).reset_index(name="count")
    return px.bar(counts, x=target_column, y="count", title="Churn Distribution")
