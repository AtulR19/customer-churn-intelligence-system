# Customer Churn Intelligence System

A modular Python project for customer churn analytics, machine learning,
batch/online inference, and a Streamlit dashboard.

## Project Structure

```text
customer-churn-intelligence-system/
├── data/                  # Raw, interim, and processed datasets
├── notebooks/             # Exploratory analysis and model experiments
├── src/
│   ├── preprocessing/     # Data loading, validation, cleaning, features
│   ├── training/          # Model training and evaluation workflows
│   ├── inference/         # Model loading and prediction utilities
│   ├── visualization/     # Reusable charting helpers
│   ├── utils/             # Shared constants and helper functions
│   └── api/               # API entry points for model serving
├── dashboard/             # Streamlit application
├── models/                # Serialized model artifacts
├── outputs/               # Reports, metrics, plots, and predictions
├── requirements.txt
├── README.md
└── main.py
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --help
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the API:

```bash
python main.py api
```

## Core Modules

- `src/preprocessing`: load customer data, clean fields, and build features.
- `src/training`: train churn models and write model artifacts.
- `src/inference`: load saved models and generate churn risk scores.
- `src/visualization`: centralize charts used by notebooks and Streamlit.
- `src/api`: expose prediction endpoints for downstream applications.
