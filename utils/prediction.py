"""
utils/prediction.py — Load model/preprocessor, run inference.
"""
import json, os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.scoring import score_lead


def _patch_sklearn_compat(obj, visited=None):
    """Recursively patch sklearn estimators to ensure cross-version compatibility.

    Fixes 'SimpleImputer' object has no attribute '_fill_dtype' which occurs
    when a model is pickled with scikit-learn 1.4.x and loaded with 1.5+.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # Walk Pipeline.named_steps
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            _patch_sklearn_compat(step, visited)

    # Walk Pipeline.steps list
    if hasattr(obj, "steps") and isinstance(obj.steps, list):
        for _, step in obj.steps:
            _patch_sklearn_compat(step, visited)

    # Walk ColumnTransformer.transformers_  (fitted)
    if hasattr(obj, "transformers_"):
        for _, transformer, _ in obj.transformers_:
            _patch_sklearn_compat(transformer, visited)

    # Walk ColumnTransformer.transformers  (unfitted)
    if hasattr(obj, "transformers") and isinstance(obj.transformers, list):
        for _, transformer, _ in obj.transformers:
            _patch_sklearn_compat(transformer, visited)

    # Patch SimpleImputer missing _fill_dtype (sklearn 1.5+ compatibility)
    if type(obj).__name__ == "SimpleImputer":
        if not hasattr(obj, "_fill_dtype"):
            stats = getattr(obj, "statistics_", None)
            obj._fill_dtype = getattr(stats, "dtype", np.float64)

    # Patch OneHotEncoder: sparse_output renamed from sparse in sklearn 1.2
    if type(obj).__name__ == "OneHotEncoder":
        if not hasattr(obj, "sparse_output") and hasattr(obj, "sparse"):
            obj.sparse_output = obj.sparse
        elif not hasattr(obj, "sparse") and hasattr(obj, "sparse_output"):
            obj.sparse = obj.sparse_output


def load_model():
    """Load the trained classifier."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run pipeline/train.py first."
        )
    return joblib.load(config.MODEL_PATH)


def load_preprocessor():
    """Load the fitted preprocessing pipeline and apply cross-version compatibility patches."""
    if not os.path.exists(config.PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found at {config.PREPROCESSOR_PATH}. Run pipeline/train.py first."
        )
    prep = joblib.load(config.PREPROCESSOR_PATH)
    _patch_sklearn_compat(prep)
    return prep


def load_metadata():
    """Load model metadata JSON."""
    if not os.path.exists(config.METADATA_PATH):
        return {}
    with open(config.METADATA_PATH) as f:
        return json.load(f)


def _get_expected_columns(preprocessor):
    """Extract columns the pipeline's inner ColumnTransformer expects."""
    try:
        # Pipeline structure: feature_engineer → preprocessor → column_transformer
        inner = preprocessor.named_steps.get("preprocessor", preprocessor)
        ct = inner.named_steps["column_transformer"]
        cols = []
        for _, _, feature_cols in ct.transformers_:
            if isinstance(feature_cols, list):
                cols.extend(feature_cols)
        return cols
    except Exception:
        return []


def align_columns(df, preprocessor):
    """Pad missing columns with NaN so the pipeline doesn't error."""
    df = df.copy()
    for col in _get_expected_columns(preprocessor):
        if col not in df.columns:
            df[col] = np.nan
    return df


def predict_single(input_df, preprocessor, model):
    """Run inference on a one-row DataFrame.
    Returns (prediction: int, probability: float).
    """
    aligned = align_columns(input_df, preprocessor)
    X_t = preprocessor.transform(aligned)
    prediction = int(model.predict(X_t)[0])
    probability = float(model.predict_proba(X_t)[0, 1])
    return prediction, probability
