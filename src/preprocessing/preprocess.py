"""Reusable preprocessing pipeline for the IBM Telco churn dataset."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_loader import load_customer_data


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
DEFAULT_TARGET_ENCODER_PATH = PROJECT_ROOT / "models" / "target_encoder.joblib"

DEFAULT_TARGET_COLUMN = "Churn"
DEFAULT_ID_COLUMNS = ("customerID",)
TELCO_CATEGORICAL_OVERRIDES = ("SeniorCitizen",)


@dataclass(frozen=True)
class PreprocessingResult:
    """Container returned by the end-to-end preprocessing workflow."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: Pipeline
    target_encoder: LabelEncoder
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    train_index: pd.Index
    test_index: pd.Index


def clean_telco_dataframe(
    data: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    id_columns: Iterable[str] = DEFAULT_ID_COLUMNS,
) -> pd.DataFrame:
    """Clean the raw Telco dataframe before splitting.

    The function removes duplicate rows, removes duplicate customer IDs when
    present, standardizes blank strings to missing values, converts
    ``TotalCharges`` to numeric, and drops rows with a missing target.
    """
    cleaned = data.copy()

    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    cleaned = cleaned.replace("", np.nan)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    cleaned = cleaned.drop_duplicates()

    existing_id_columns = [column for column in id_columns if column in cleaned.columns]
    if existing_id_columns:
        cleaned = cleaned.drop_duplicates(subset=existing_id_columns, keep="first")

    if target_column in cleaned.columns:
        cleaned = cleaned.dropna(subset=[target_column])

    return cleaned.reset_index(drop=True)


def split_features_target(
    data: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Iterable[str] = DEFAULT_ID_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into feature and target columns."""
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataframe.")

    removable_columns = [target_column, *[column for column in drop_columns if column in data.columns]]
    features = data.drop(columns=removable_columns)
    target = data[target_column]

    return features, target


def label_encode_target(target: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode a text target column, e.g. ``No``/``Yes``, as numeric labels."""
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target.astype(str))
    return encoded_target, encoder


def infer_feature_types(
    features: pd.DataFrame,
    *,
    categorical_overrides: Iterable[str] = TELCO_CATEGORICAL_OVERRIDES,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns for the preprocessing pipeline."""
    categorical_features = list(features.select_dtypes(include=["object", "string", "category", "bool"]).columns)

    for column in categorical_overrides:
        if column in features.columns and column not in categorical_features:
            categorical_features.append(column)

    numeric_features = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in categorical_features
    ]

    return numeric_features, categorical_features


def build_preprocessing_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> Pipeline:
    """Build the scikit-learn preprocessing pipeline.

    Numeric columns are median-imputed and scaled. Categorical columns are
    mode-imputed and one-hot encoded with unknown-category protection.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(numeric_features)),
            ("categorical", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[("preprocessor", column_transformer)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return transformed feature names from a fitted preprocessing pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def transform_to_dataframe(
    pipeline: Pipeline,
    features: pd.DataFrame,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Transform features and return a dataframe with transformed feature names."""
    transformed = pipeline.transform(features)
    feature_names = get_feature_names(pipeline)
    return pd.DataFrame(transformed, columns=feature_names, index=index)


def save_preprocessing_pipeline(pipeline: Pipeline, path: str | Path = DEFAULT_PIPELINE_PATH) -> Path:
    """Persist the fitted preprocessing pipeline to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def save_target_encoder(encoder: LabelEncoder, path: str | Path = DEFAULT_TARGET_ENCODER_PATH) -> Path:
    """Persist the fitted target label encoder to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, output_path)
    return output_path


def load_preprocessing_pipeline(path: str | Path = DEFAULT_PIPELINE_PATH) -> Pipeline:
    """Load a saved preprocessing pipeline."""
    return joblib.load(path)


def preprocess_train_test_split(
    data: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Iterable[str] = DEFAULT_ID_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    pipeline_output_path: str | Path | None = DEFAULT_PIPELINE_PATH,
    target_encoder_output_path: str | Path | None = DEFAULT_TARGET_ENCODER_PATH,
) -> PreprocessingResult:
    """Clean, encode, split, fit, transform, and optionally save preprocessing artifacts."""
    cleaned = clean_telco_dataframe(data, target_column=target_column, id_columns=drop_columns)
    features, target = split_features_target(
        cleaned,
        target_column=target_column,
        drop_columns=drop_columns,
    )
    encoded_target, target_encoder = label_encode_target(target)
    numeric_features, categorical_features = infer_feature_types(features)

    stratify_values = encoded_target if stratify else None
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
    pipeline.fit(x_train_raw)

    x_train = transform_to_dataframe(pipeline, x_train_raw, index=x_train_raw.index)
    x_test = transform_to_dataframe(pipeline, x_test_raw, index=x_test_raw.index)
    feature_names = list(x_train.columns)

    if pipeline_output_path is not None:
        save_preprocessing_pipeline(pipeline, pipeline_output_path)

    if target_encoder_output_path is not None:
        save_target_encoder(target_encoder, target_encoder_output_path)

    return PreprocessingResult(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
        target_encoder=target_encoder,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        train_index=x_train_raw.index,
        test_index=x_test_raw.index,
    )


def preprocess_telco_churn_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    **kwargs,
) -> PreprocessingResult:
    """Load the IBM Telco churn CSV and run the full preprocessing workflow."""
    data = load_customer_data(data_path)
    return preprocess_train_test_split(data, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command line parser for running preprocessing directly."""
    parser = argparse.ArgumentParser(description="Preprocess the IBM Telco churn dataset.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to the raw Telco churn CSV.")
    parser.add_argument(
        "--pipeline-output-path",
        default=str(DEFAULT_PIPELINE_PATH),
        help="Where to save the fitted preprocessing pipeline.",
    )
    parser.add_argument(
        "--target-encoder-output-path",
        default=str(DEFAULT_TARGET_ENCODER_PATH),
        help="Where to save the fitted target label encoder.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train-test split.")
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified train-test splitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run preprocessing from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = preprocess_telco_churn_dataset(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=not args.no_stratify,
        pipeline_output_path=args.pipeline_output_path,
        target_encoder_output_path=args.target_encoder_output_path,
    )

    print(f"Training rows: {result.x_train.shape[0]:,}")
    print(f"Test rows: {result.x_test.shape[0]:,}")
    print(f"Transformed features: {result.x_train.shape[1]:,}")
    print(f"Numeric features scaled: {len(result.numeric_features):,}")
    print(f"Categorical features encoded: {len(result.categorical_features):,}")
    print(f"Pipeline saved to: {Path(args.pipeline_output_path).resolve()}")
    print(f"Target encoder saved to: {Path(args.target_encoder_output_path).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
