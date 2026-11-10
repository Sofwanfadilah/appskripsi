"""Public machine-learning API for the FMA Small genre model."""

from genre_model import (
    MODEL_PATH,
    DATASET_DIR,
    DATASET_AUDIT_PATH,
    DATASET_SPECS,
    FINAL_DATASET_PATH,
    build_final_dataset,
    extract_audio_features,
    extract_fma_metadata,
    get_feature_columns,
    load_model,
    predict_genre,
    train_from_fma,
)

__all__ = [
    "MODEL_PATH",
    "DATASET_DIR",
    "DATASET_AUDIT_PATH",
    "DATASET_SPECS",
    "FINAL_DATASET_PATH",
    "build_final_dataset",
    "extract_audio_features",
    "extract_fma_metadata",
    "get_feature_columns",
    "load_model",
    "predict_genre",
    "train_from_fma",
]
