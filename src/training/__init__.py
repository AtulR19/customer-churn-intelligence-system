"""Training workflows for churn prediction models."""

__all__ = [
    "ModelEvaluation",
    "TrainingResult",
    "build_candidate_models",
    "build_model_comparison",
    "calculate_scale_pos_weight",
    "evaluate_model",
    "load_best_model",
    "save_best_model",
    "save_confusion_matrices",
    "save_model_comparison",
    "select_best_model",
    "train_and_evaluate_models",
    "train_churn_model",
    "train_churn_models",
]


def __getattr__(name: str):
    """Lazily expose training functions without eager module loading."""
    if name in __all__:
        from src.training import train_model

        return getattr(train_model, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
