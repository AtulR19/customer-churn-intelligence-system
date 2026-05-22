"""Preprocessing utilities for churn data."""

__all__ = [
    "PreprocessingResult",
    "build_preprocessing_pipeline",
    "clean_telco_dataframe",
    "get_feature_names",
    "infer_feature_types",
    "label_encode_target",
    "load_preprocessing_pipeline",
    "preprocess_telco_churn_dataset",
    "preprocess_train_test_split",
    "save_preprocessing_pipeline",
    "save_target_encoder",
    "split_features_target",
    "transform_to_dataframe",
]


def __getattr__(name: str):
    """Lazily expose preprocessing functions without eager module loading."""
    if name in __all__:
        from src.preprocessing import preprocess

        return getattr(preprocess, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
