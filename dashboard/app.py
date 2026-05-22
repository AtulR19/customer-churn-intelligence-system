"""Professional Streamlit dashboard for customer churn intelligence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocess import clean_telco_dataframe, split_features_target, transform_to_dataframe


DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrices.json"
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"
SHAP_SUMMARY_PATH = SHAP_DIR / "shap_summary.png"
SHAP_IMPORTANCE_PATH = SHAP_DIR / "shap_feature_importance.png"
SHAP_WATERFALL_PATH = SHAP_DIR / "shap_waterfall.png"
PREDICTION_EXPLANATION_PATH = SHAP_DIR / "prediction_explanation.json"
PREDICTION_EXPLANATION_MD_PATH = SHAP_DIR / "prediction_explanation.md"

RISK_THRESHOLD = 0.5


def configure_page() -> None:
    """Configure Streamlit and apply dashboard styling."""
    st.set_page_config(
        page_title="Customer Churn Intelligence",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --ink: #17202a;
            --muted: #667085;
            --line: #e6e8ee;
            --panel: #ffffff;
            --panel-soft: #f7f9fc;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
        }
        .stApp {
            background: #f6f8fb;
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: inherit;
        }
        section[data-testid="stSidebar"] {
            background: #111827;
        }
        section[data-testid="stSidebar"] * {
            color: #f9fafb;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #d1d5db;
        }
        .dashboard-title {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            margin: 0 0 0.15rem 0;
        }
        .dashboard-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin: 0 0 1.2rem 0;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, 0.06);
            min-height: 120px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .metric-help {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, 0.04);
        }
        .risk-panel {
            border-radius: 8px;
            border: 1px solid var(--line);
            padding: 1.1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        }
        .risk-score {
            font-size: 3rem;
            font-weight: 900;
            color: var(--ink);
            line-height: 1;
        }
        .risk-high {
            color: var(--red);
            font-weight: 800;
        }
        .risk-low {
            color: var(--green);
            font-weight: 800;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.84rem;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stButton"] > button {
            border-radius: 7px;
            border: 1px solid #1d4ed8;
            background: #1d4ed8;
            color: white;
            font-weight: 700;
            min-height: 2.75rem;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: #1e40af;
            background: #1e40af;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    """Render a consistent page header."""
    st.markdown(f"<h1 class='dashboard-title'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='dashboard-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, help_text: str) -> None:
    """Render a compact KPI card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load and lightly prepare the churn dataset for analytics."""
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["ChurnFlag"] = data["Churn"].map({"No": 0, "Yes": 1})
    data["SeniorCitizenLabel"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-0.1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    )
    data["MonthlyChargeGroup"] = pd.qcut(
        data["MonthlyCharges"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    return data


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, Any] | None:
    """Load the saved model bundle if it exists."""
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Load model comparison metrics."""
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_confusion_matrices() -> dict[str, Any]:
    """Load saved confusion matrices."""
    if not CONFUSION_MATRIX_PATH.exists():
        return {}
    return json.loads(CONFUSION_MATRIX_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_prediction_explanation() -> dict[str, Any]:
    """Load saved SHAP prediction explanation."""
    if not PREDICTION_EXPLANATION_PATH.exists():
        return {}
    return json.loads(PREDICTION_EXPLANATION_PATH.read_text(encoding="utf-8"))


def build_scored_dataset(data: pd.DataFrame, model_bundle: dict[str, Any] | None) -> pd.DataFrame:
    """Score every customer using the saved model bundle."""
    scored = data.copy()
    if model_bundle is None:
        scored["ChurnProbability"] = np.nan
        scored["RiskBand"] = "Unavailable"
        return scored

    cleaned = clean_telco_dataframe(data.drop(columns=["ChurnFlag", "SeniorCitizenLabel", "TenureGroup", "MonthlyChargeGroup"], errors="ignore"))
    raw_features, _ = split_features_target(cleaned)
    transformed = transform_to_dataframe(model_bundle["preprocessing_pipeline"], raw_features, index=cleaned.index)
    probabilities = model_bundle["model"].predict_proba(transformed)[:, 1]

    scored.loc[cleaned.index, "ChurnProbability"] = probabilities
    scored["RiskBand"] = pd.cut(
        scored["ChurnProbability"],
        bins=[-0.01, 0.3, 0.5, 1.0],
        labels=["Low", "Moderate", "High"],
    ).astype("string")
    return scored


def churn_rate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate customer count and churn rate by a categorical field."""
    grouped = (
        data.groupby(column, observed=True)
        .agg(Customers=("customerID", "count"), ChurnRate=("ChurnFlag", "mean"))
        .reset_index()
    )
    grouped["ChurnRatePct"] = grouped["ChurnRate"] * 100
    return grouped


def sidebar_navigation() -> str:
    """Render sidebar navigation."""
    st.sidebar.markdown("## Customer Churn Intelligence")
    st.sidebar.caption("Prediction, analytics, and explainability")
    st.sidebar.divider()
    return st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Churn Analytics",
            "Predict Churn",
            "SHAP Explainability",
            "Model Comparison",
        ],
        label_visibility="collapsed",
    )


def render_executive_overview(data: pd.DataFrame, scored: pd.DataFrame, model_bundle: dict[str, Any] | None) -> None:
    """Render top-level operational KPIs and charts."""
    render_header(
        "Customer Churn Intelligence",
        "Executive view of churn exposure, model risk scores, and customer retention signals.",
    )

    total_customers = len(data)
    churned_customers = int(data["ChurnFlag"].sum())
    churn_rate = data["ChurnFlag"].mean()
    avg_monthly_charge = data["MonthlyCharges"].mean()
    at_risk_count = int((scored["ChurnProbability"] >= RISK_THRESHOLD).sum()) if scored["ChurnProbability"].notna().any() else 0

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card("Customers", f"{total_customers:,}", "Total records in the IBM Telco dataset")
    with kpi_cols[1]:
        render_kpi_card("Observed Churn", f"{churn_rate:.1%}", f"{churned_customers:,} customers churned")
    with kpi_cols[2]:
        render_kpi_card("Predicted At Risk", f"{at_risk_count:,}", f"Risk threshold at {RISK_THRESHOLD:.0%}")
    with kpi_cols[3]:
        render_kpi_card("Avg Monthly Charge", f"${avg_monthly_charge:,.2f}", "Mean monthly customer revenue")

    st.write("")

    left, right = st.columns((1.1, 1))
    with left:
        contract = churn_rate_by(data, "Contract").sort_values("ChurnRatePct", ascending=False)
        fig = px.bar(
            contract,
            x="Contract",
            y="ChurnRatePct",
            color="Contract",
            text="ChurnRatePct",
            title="Churn Rate by Contract Type",
            color_discrete_sequence=["#2563eb", "#0891b2", "#16a34a"],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)", xaxis_title=None, height=390)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        risk_counts = scored["RiskBand"].value_counts(dropna=False).rename_axis("RiskBand").reset_index(name="Customers")
        fig = px.pie(
            risk_counts,
            names="RiskBand",
            values="Customers",
            hole=0.52,
            title="Predicted Risk Mix",
            color="RiskBand",
            color_discrete_map={"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626", "Unavailable": "#667085"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        tenure = churn_rate_by(data, "TenureGroup")
        fig = px.line(
            tenure,
            x="TenureGroup",
            y="ChurnRatePct",
            markers=True,
            title="Churn Rate by Tenure Group",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_traces(line_width=3)
        fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="Tenure in months", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with lower_right:
        if model_bundle:
            metrics = model_bundle.get("metrics", {})
            metric_cols = st.columns(3)
            metric_cols[0].metric("Best Model", model_bundle.get("model_name", "Unknown"))
            metric_cols[1].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
        top_segments = churn_rate_by(data, "PaymentMethod").sort_values("ChurnRatePct", ascending=False)
        fig = px.bar(
            top_segments,
            x="ChurnRatePct",
            y="PaymentMethod",
            orientation="h",
            title="Churn Rate by Payment Method",
            color="ChurnRatePct",
            color_continuous_scale=["#dbeafe", "#2563eb"],
        )
        fig.update_layout(xaxis_title="Churn rate (%)", yaxis_title=None, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def render_churn_analytics(data: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Render detailed churn analytics."""
    render_header(
        "Churn Analytics",
        "Segment-level churn patterns across tenure, contracts, billing, services, and charges.",
    )

    filter_cols = st.columns(4)
    contract_filter = filter_cols[0].multiselect("Contract", sorted(data["Contract"].dropna().unique()))
    internet_filter = filter_cols[1].multiselect("Internet service", sorted(data["InternetService"].dropna().unique()))
    churn_filter = filter_cols[2].multiselect("Churn status", sorted(data["Churn"].dropna().unique()))
    risk_filter = filter_cols[3].multiselect("Risk band", sorted(scored["RiskBand"].dropna().unique()))

    filtered = scored.copy()
    if contract_filter:
        filtered = filtered[filtered["Contract"].isin(contract_filter)]
    if internet_filter:
        filtered = filtered[filtered["InternetService"].isin(internet_filter)]
    if churn_filter:
        filtered = filtered[filtered["Churn"].isin(churn_filter)]
    if risk_filter:
        filtered = filtered[filtered["RiskBand"].isin(risk_filter)]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Filtered Customers", f"{len(filtered):,}")
    metric_cols[1].metric("Churn Rate", f"{filtered['ChurnFlag'].mean():.1%}" if len(filtered) else "0.0%")
    metric_cols[2].metric("Avg Tenure", f"{filtered['tenure'].mean():.1f} mo" if len(filtered) else "0.0 mo")
    metric_cols[3].metric("Avg Risk", f"{filtered['ChurnProbability'].mean():.1%}" if filtered["ChurnProbability"].notna().any() else "N/A")

    first_row = st.columns(2)
    with first_row[0]:
        churn_counts = filtered["Churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
        fig = px.bar(churn_counts, x="Churn", y="Customers", color="Churn", title="Churn Distribution")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with first_row[1]:
        fig = px.histogram(
            filtered,
            x="tenure",
            color="Churn",
            nbins=36,
            barmode="overlay",
            opacity=0.72,
            title="Tenure Distribution by Churn",
        )
        fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers", height=380)
        st.plotly_chart(fig, use_container_width=True)

    second_row = st.columns(2)
    with second_row[0]:
        fig = px.box(
            filtered,
            x="Contract",
            y="MonthlyCharges",
            color="Churn",
            points="outliers",
            title="Monthly Charges by Contract and Churn",
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Monthly charges", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with second_row[1]:
        internet = churn_rate_by(filtered, "InternetService").sort_values("ChurnRatePct", ascending=False)
        fig = px.bar(
            internet,
            x="InternetService",
            y="ChurnRatePct",
            color="InternetService",
            text="ChurnRatePct",
            title="Churn Rate by Internet Service",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)", xaxis_title=None, height=420)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        size="ChurnProbability",
        hover_data=["customerID", "Contract", "InternetService", "PaymentMethod", "ChurnProbability"],
        title="Customer Map: Tenure, Charges, and Predicted Risk",
    )
    fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Monthly charges", height=520)
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_form(model_bundle: dict[str, Any] | None) -> None:
    """Render single-customer prediction form."""
    render_header(
        "Predict Churn",
        "Score an individual customer with the trained churn model.",
    )

    if model_bundle is None:
        st.error("Best model artifact not found. Run `python main.py train` first.")
        return

    left, right = st.columns((1.2, 0.8))
    with left:
        with st.form("prediction_form"):
            st.subheader("Customer Profile")
            row_1 = st.columns(3)
            gender = row_1[0].selectbox("Gender", ["Female", "Male"])
            senior_citizen = row_1[1].selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            partner = row_1[2].selectbox("Partner", ["No", "Yes"])

            row_2 = st.columns(3)
            dependents = row_2[0].selectbox("Dependents", ["No", "Yes"])
            tenure = row_2[1].number_input("Tenure", min_value=0, max_value=72, value=12, step=1)
            contract = row_2[2].selectbox("Contract", ["Month-to-month", "One year", "Two year"])

            row_3 = st.columns(3)
            phone_service = row_3[0].selectbox("Phone service", ["No", "Yes"], index=1)
            multiple_lines = row_3[1].selectbox("Multiple lines", ["No", "No phone service", "Yes"])
            internet_service = row_3[2].selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=1)

            row_4 = st.columns(3)
            online_security = row_4[0].selectbox("Online security", ["No", "No internet service", "Yes"])
            online_backup = row_4[1].selectbox("Online backup", ["No", "No internet service", "Yes"])
            device_protection = row_4[2].selectbox("Device protection", ["No", "No internet service", "Yes"])

            row_5 = st.columns(3)
            tech_support = row_5[0].selectbox("Tech support", ["No", "No internet service", "Yes"])
            streaming_tv = row_5[1].selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            streaming_movies = row_5[2].selectbox("Streaming movies", ["No", "No internet service", "Yes"])

            row_6 = st.columns(3)
            paperless_billing = row_6[0].selectbox("Paperless billing", ["No", "Yes"], index=1)
            payment_method = row_6[1].selectbox(
                "Payment method",
                ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                index=2,
            )
            monthly_charges = row_6[2].number_input("Monthly charges", min_value=0.0, max_value=200.0, value=85.0, step=1.0)

            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(max(tenure, 1) * monthly_charges),
                step=10.0,
            )
            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    with right:
        st.subheader("Prediction Result")
        if submitted:
            input_data = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                    }
                ]
            )
            transformed = transform_to_dataframe(model_bundle["preprocessing_pipeline"], input_data, index=input_data.index)
            probability = float(model_bundle["model"].predict_proba(transformed)[0, 1])
            prediction = "Yes" if probability >= RISK_THRESHOLD else "No"
            risk_class = "risk-high" if prediction == "Yes" else "risk-low"
            gauge_color = "#dc2626" if prediction == "Yes" else "#16a34a"

            st.markdown(
                f"""
                <div class="risk-panel">
                    <div class="metric-label">Churn Probability</div>
                    <div class="risk-score">{probability:.1%}</div>
                    <div class="{risk_class}">Predicted churn: {prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 30], "color": "#dcfce7"},
                            {"range": [30, 50], "color": "#fef3c7"},
                            {"range": [50, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {"line": {"color": "#111827", "width": 3}, "value": RISK_THRESHOLD * 100},
                    },
                )
            )
            gauge.update_layout(height=290, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(gauge, use_container_width=True)

            st.dataframe(input_data.T.rename(columns={0: "value"}), use_container_width=True)
        else:
            st.markdown(
                """
                <div class="risk-panel">
                    <div class="metric-label">Churn Probability</div>
                    <div class="risk-score">--</div>
                    <div class="small-note">Submit a profile to score churn risk.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_shap_page() -> None:
    """Render saved SHAP explainability artifacts."""
    render_header(
        "SHAP Explainability",
        "Global feature impact and local prediction rationale for the trained churn model.",
    )

    missing = [
        path
        for path in [SHAP_SUMMARY_PATH, SHAP_IMPORTANCE_PATH, SHAP_WATERFALL_PATH, PREDICTION_EXPLANATION_PATH]
        if not path.exists()
    ]
    if missing:
        st.warning("SHAP artifacts are missing. Run `python main.py explain` to generate them.")
        return

    explanation = load_prediction_explanation()

    top_cols = st.columns(4)
    top_cols[0].metric("Explained Model", explanation.get("model_name", "Unknown"))
    top_cols[1].metric("Customer ID", explanation.get("customer_id", "N/A"))
    top_cols[2].metric("Predicted Churn", explanation.get("predicted_label", "N/A"))
    top_cols[3].metric("Probability", f"{explanation.get('predicted_probability_churn', 0):.1%}")

    tabs = st.tabs(["Global Summary", "Feature Importance", "Waterfall", "Prediction Explanation"])
    with tabs[0]:
        st.image(str(SHAP_SUMMARY_PATH), use_container_width=True)
    with tabs[1]:
        st.image(str(SHAP_IMPORTANCE_PATH), use_container_width=True)
    with tabs[2]:
        st.image(str(SHAP_WATERFALL_PATH), use_container_width=True)
    with tabs[3]:
        if PREDICTION_EXPLANATION_MD_PATH.exists():
            st.markdown(PREDICTION_EXPLANATION_MD_PATH.read_text(encoding="utf-8"))
        contributions = pd.DataFrame(explanation.get("top_features_by_absolute_impact", []))
        if not contributions.empty:
            fig = px.bar(
                contributions.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                color="direction",
                title="Top Local Feature Contributions",
                color_discrete_map={
                    "increases churn risk": "#dc2626",
                    "decreases churn risk": "#16a34a",
                },
            )
            fig.update_layout(xaxis_title="SHAP value", yaxis_title=None, height=460)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(contributions, use_container_width=True, hide_index=True)


def render_model_comparison() -> None:
    """Render model metrics and confusion matrices."""
    render_header(
        "Model Comparison",
        "Performance comparison across Logistic Regression, Random Forest, and XGBoost.",
    )

    comparison = load_model_comparison()
    matrices = load_confusion_matrices()

    if comparison.empty:
        st.warning("Model comparison output not found. Run `python main.py train` first.")
        return

    best_row = comparison.sort_values(["roc_auc", "f1_score", "recall"], ascending=False).iloc[0]
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Best Model", best_row["model"])
    kpi_cols[1].metric("Accuracy", f"{best_row['accuracy']:.3f}")
    kpi_cols[2].metric("Precision", f"{best_row['precision']:.3f}")
    kpi_cols[3].metric("Recall", f"{best_row['recall']:.3f}")
    kpi_cols[4].metric("ROC-AUC", f"{best_row['roc_auc']:.3f}")

    st.subheader("Metric Comparison")
    metric_long = comparison.melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        metric_long,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        title="Model Metrics",
    )
    fig.update_layout(yaxis_title="Score", xaxis_title=None, height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparison.style.format({column: "{:.4f}" for column in comparison.columns if column != "model"}), use_container_width=True)

    st.subheader("Confusion Matrices")
    selected_model = st.selectbox("Model", comparison["model"].tolist())
    matrix_payload = matrices.get(selected_model)
    if matrix_payload:
        matrix = np.array(matrix_payload["matrix"])
        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            x=["No churn", "Churn"],
            y=["No churn", "Churn"],
            title=f"{selected_model} Confusion Matrix",
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No confusion matrix found for the selected model.")


def main() -> None:
    """Run the Streamlit dashboard."""
    configure_page()
    data = load_raw_data()
    model_bundle = load_model_bundle()
    scored = build_scored_dataset(data, model_bundle)

    page = sidebar_navigation()

    if page == "Executive Overview":
        render_executive_overview(data, scored, model_bundle)
    elif page == "Churn Analytics":
        render_churn_analytics(data, scored)
    elif page == "Predict Churn":
        render_prediction_form(model_bundle)
    elif page == "SHAP Explainability":
        render_shap_page()
    elif page == "Model Comparison":
        render_model_comparison()


if __name__ == "__main__":
    main()
