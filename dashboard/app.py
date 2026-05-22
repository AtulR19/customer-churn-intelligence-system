"""Streamlit dashboard for customer churn intelligence."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Intelligence",
        layout="wide",
    )

    st.title("Customer Churn Intelligence")

    left, middle, right = st.columns(3)
    left.metric("Customers", "0")
    middle.metric("At Risk", "0")
    right.metric("Churn Rate", "0.0%")

    st.divider()
    st.subheader("Churn Overview")
    st.info("Connect data and trained model artifacts to populate this dashboard.")


if __name__ == "__main__":
    main()
