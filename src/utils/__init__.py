"""Shared utilities for the churn intelligence project."""

__all__ = [
    "ExplainabilityData",
    "ShapArtifacts",
    "build_prediction_explanation",
    "build_shap_explainer",
    "compute_shap_values",
    "load_model_bundle",
    "prepare_explainability_data",
    "run_shap_explainability",
    "save_feature_importance_plot",
    "save_prediction_explanation",
    "save_summary_plot",
    "save_waterfall_plot",
]


def __getattr__(name: str):
    """Lazily expose utility functions without eager imports."""
    if name in __all__:
        from src.utils import explainability

        return getattr(explainability, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
