"""Complete model training pipeline for customer churn prediction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_loader import load_customer_data
from src.preprocessing.preprocess import (
    DEFAULT_DATA_PATH,
    DEFAULT_ID_COLUMNS,
    DEFAULT_PIPELINE_PATH,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TARGET_ENCODER_PATH,
    PreprocessingResult,
    preprocess_train_test_split,
)


DEFAULT_BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.joblib"
DEFAULT_MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrices.json"


@dataclass(frozen=True)
class ModelEvaluation:
    """Evaluation output for one fitted churn model."""

    model_name: str
    model: BaseEstimator
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class TrainingResult:
    """Container returned by the full training workflow."""

    preprocessing: PreprocessingResult
    evaluations: list[ModelEvaluation]
    comparison: pd.DataFrame
    best_evaluation: ModelEvaluation
    best_model_path: Path
    comparison_path: Path
    confusion_matrix_path: Path


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """Calculate XGBoost class imbalance weight as negative / positive."""
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))

    if positive_count == 0:
        return 1.0

    return negative_count / positive_count


def build_candidate_models(
    *,
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> dict[str, BaseEstimator]:
    """Create the candidate churn classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def predict_positive_class_probability(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for metrics such as ROC-AUC."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not support predict_proba.")

    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> ModelEvaluation:
    """Evaluate a fitted model with classification metrics and confusion matrix."""
    predictions = model.predict(x_test)
    probabilities = predict_positive_class_probability(model, x_test)

    try:
        roc_auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": roc_auc,
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).astype(int).tolist()

    return ModelEvaluation(
        model_name=model_name,
        model=model,
        metrics=metrics,
        confusion_matrix=matrix,
    )


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = 42,
) -> list[ModelEvaluation]:
    """Train all candidate models and return their evaluations."""
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    candidates = build_candidate_models(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )

    evaluations: list[ModelEvaluation] = []
    for model_name, model in candidates.items():
        model.fit(x_train, y_train)
        evaluations.append(evaluate_model(model_name, model, x_test, y_test))

    return evaluations


def build_model_comparison(evaluations: Iterable[ModelEvaluation]) -> pd.DataFrame:
    """Create a sorted model comparison dataframe."""
    rows = [
        {
            "model": evaluation.model_name,
            **evaluation.metrics,
        }
        for evaluation in evaluations
    ]
    comparison = pd.DataFrame(rows)
    return comparison.sort_values(
        by=["roc_auc", "f1_score", "recall"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def select_best_model(evaluations: Iterable[ModelEvaluation]) -> ModelEvaluation:
    """Select the best model using ROC-AUC, then F1-score, then recall."""
    return max(
        evaluations,
        key=lambda evaluation: (
            np.nan_to_num(evaluation.metrics["roc_auc"], nan=-1.0),
            evaluation.metrics["f1_score"],
            evaluation.metrics["recall"],
        ),
    )


def save_model_comparison(
    comparison: pd.DataFrame,
    path: str | Path = DEFAULT_MODEL_COMPARISON_PATH,
) -> Path:
    """Save the model comparison metrics as a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def save_confusion_matrices(
    evaluations: Iterable[ModelEvaluation],
    path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
) -> Path:
    """Save confusion matrices for all trained models as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    for evaluation in evaluations:
        tn, fp = evaluation.confusion_matrix[0]
        fn, tp = evaluation.confusion_matrix[1]
        payload[evaluation.model_name] = {
            "labels": {"0": "No churn", "1": "Churn"},
            "matrix": evaluation.confusion_matrix,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_best_model(
    best_evaluation: ModelEvaluation,
    preprocessing: PreprocessingResult,
    path: str | Path = DEFAULT_BEST_MODEL_PATH,
) -> Path:
    """Save the best trained model and preprocessing assets as a joblib bundle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model_name": best_evaluation.model_name,
        "model": best_evaluation.model,
        "metrics": best_evaluation.metrics,
        "confusion_matrix": best_evaluation.confusion_matrix,
        "preprocessing_pipeline": preprocessing.pipeline,
        "target_encoder": preprocessing.target_encoder,
        "target_classes": preprocessing.target_encoder.classes_.tolist(),
        "feature_names": preprocessing.feature_names,
        "numeric_features": preprocessing.numeric_features,
        "categorical_features": preprocessing.categorical_features,
    }
    joblib.dump(model_bundle, output_path)

    return output_path


def load_best_model(path: str | Path = DEFAULT_BEST_MODEL_PATH) -> dict:
    """Load a saved best-model bundle."""
    return joblib.load(path)


def train_churn_models(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_columns: Iterable[str] = DEFAULT_ID_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    best_model_output_path: str | Path = DEFAULT_BEST_MODEL_PATH,
    comparison_output_path: str | Path = DEFAULT_MODEL_COMPARISON_PATH,
    confusion_matrix_output_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    preprocessing_pipeline_output_path: str | Path | None = DEFAULT_PIPELINE_PATH,
    target_encoder_output_path: str | Path | None = DEFAULT_TARGET_ENCODER_PATH,
) -> TrainingResult:
    """Run the complete churn model training and evaluation workflow."""
    raw_data = load_customer_data(data_path)
    preprocessing = preprocess_train_test_split(
        raw_data,
        target_column=target_column,
        drop_columns=drop_columns,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
        pipeline_output_path=preprocessing_pipeline_output_path,
        target_encoder_output_path=target_encoder_output_path,
    )

    evaluations = train_and_evaluate_models(
        preprocessing.x_train,
        preprocessing.x_test,
        preprocessing.y_train,
        preprocessing.y_test,
        random_state=random_state,
    )
    comparison = build_model_comparison(evaluations)
    best_evaluation = select_best_model(evaluations)

    comparison_path = save_model_comparison(comparison, comparison_output_path)
    confusion_matrix_path = save_confusion_matrices(evaluations, confusion_matrix_output_path)
    best_model_path = save_best_model(best_evaluation, preprocessing, best_model_output_path)

    return TrainingResult(
        preprocessing=preprocessing,
        evaluations=evaluations,
        comparison=comparison,
        best_evaluation=best_evaluation,
        best_model_path=best_model_path,
        comparison_path=comparison_path,
        confusion_matrix_path=confusion_matrix_path,
    )


def train_churn_model(
    data: pd.DataFrame,
    target_column: str,
    model_path: str | Path,
) -> dict[str, float]:
    """Backward-compatible helper that trains all models and saves the best one."""
    preprocessing = preprocess_train_test_split(
        data,
        target_column=target_column,
        pipeline_output_path=None,
        target_encoder_output_path=None,
    )
    evaluations = train_and_evaluate_models(
        preprocessing.x_train,
        preprocessing.x_test,
        preprocessing.y_train,
        preprocessing.y_test,
    )
    best_evaluation = select_best_model(evaluations)
    save_best_model(best_evaluation, preprocessing, model_path)

    return best_evaluation.metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to the raw Telco churn CSV.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train-test split and models.")
    parser.add_argument(
        "--best-model-output-path",
        default=str(DEFAULT_BEST_MODEL_PATH),
        help="Where to save the best model joblib bundle.",
    )
    parser.add_argument(
        "--comparison-output-path",
        default=str(DEFAULT_MODEL_COMPARISON_PATH),
        help="Where to save model comparison metrics.",
    )
    parser.add_argument(
        "--confusion-matrix-output-path",
        default=str(DEFAULT_CONFUSION_MATRIX_PATH),
        help="Where to save confusion matrices.",
    )
    return parser


def print_training_summary(result: TrainingResult) -> None:
    """Print a concise training summary."""
    print("\nModel comparison:")
    print(result.comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model: {result.best_evaluation.model_name}")
    print(f"Best model saved to: {result.best_model_path.resolve()}")
    print(f"Model comparison saved to: {result.comparison_path.resolve()}")
    print(f"Confusion matrices saved to: {result.confusion_matrix_path.resolve()}")


def main(argv: list[str] | None = None) -> int:
    """Run training from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = train_churn_models(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        best_model_output_path=args.best_model_output_path,
        comparison_output_path=args.comparison_output_path,
        confusion_matrix_output_path=args.confusion_matrix_output_path,
    )
    print_training_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
