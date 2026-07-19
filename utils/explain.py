"""
utils/explain.py — SHAP-based explainability utilities.

Dynamically selects the correct SHAP explainer based on the
fitted model type — TreeExplainer for tree/boosting models,
LinearExplainer for Logistic Regression.
"""

import logging
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Models that support TreeExplainer
TREE_MODEL_TYPES = {
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "DecisionTreeClassifier",
    "AdaBoostClassifier",
    "ExtraTreesClassifier",
}


def get_shap_explainer(model, X_background: np.ndarray):
    """Dynamically select and return the correct SHAP explainer.

    Selection logic:
        - RandomForest, GradientBoosting, XGBoost, DecisionTree,
          AdaBoost, ExtraTrees → shap.TreeExplainer
        - LogisticRegression → shap.LinearExplainer

    Args:
        model: Fitted sklearn/XGBoost classifier.
        X_background: Background dataset (training set sample) used
                      for LinearExplainer reference distribution.

    Returns:
        Configured SHAP explainer instance.

    Raises:
        ImportError: If shap is not installed.
        TypeError: If the model type is not supported.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is not installed. Run: pip install shap") from exc

    model_type = type(model).__name__
    logger.info("Selecting SHAP explainer for model type: %s", model_type)

    if model_type in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        logger.info("Using TreeExplainer for %s", model_type)
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_background)
        logger.info("Using LinearExplainer for %s", model_type)
    else:
        # Fallback: use KernelExplainer with a small background sample
        logger.warning(
            "Unknown model type '%s' — falling back to KernelExplainer (slow).",
            model_type,
        )
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for given instances.

    For binary classification, returns values for the positive class (index 1).

    Args:
        explainer: Fitted SHAP explainer.
        X: Feature matrix (n_samples, n_features).

    Returns:
        SHAP values array of shape (n_samples, n_features).
    """
    try:
        shap_vals = explainer.shap_values(X)
        # For binary classifiers, shap_values returns a list of 2 arrays
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]  # Positive class
        return np.array(shap_vals)
    except Exception as exc:
        logger.error("Failed to compute SHAP values: %s", exc)
        raise


def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    n: int = 10,
) -> pd.DataFrame:
    """Extract the top contributing features for a single instance.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: List of feature names corresponding to columns.
        instance_idx: Index of the instance to explain.
        n: Number of top features to return.

    Returns:
        DataFrame with columns: feature, shap_value, abs_value, direction.
        Sorted by abs_value descending.
    """
    if instance_idx >= len(shap_values):
        raise IndexError(
            f"instance_idx={instance_idx} out of range for {len(shap_values)} samples."
        )

    instance_shap = shap_values[instance_idx]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": instance_shap,
    })
    df["abs_value"] = df["shap_value"].abs()
    df["direction"] = df["shap_value"].apply(
        lambda v: "positive" if v >= 0 else "negative"
    )
    return df.nlargest(n, "abs_value").reset_index(drop=True)


def plot_waterfall(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    title: str = "SHAP Feature Contributions",
) -> go.Figure:
    """Create a waterfall bar chart showing SHAP contributions for one instance.

    Positive contributions (push toward conversion) are shown in green,
    negative contributions (push against conversion) in red.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: Feature names list.
        instance_idx: Which instance to plot.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    top_df = get_top_features(shap_values, feature_names, instance_idx, n=15)

    colors = [
        "#10b981" if v >= 0 else "#ef4444"
        for v in top_df["shap_value"]
    ]

    fig = go.Figure(go.Bar(
        x=top_df["shap_value"],
        y=top_df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_df["shap_value"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    title: str = "Global Feature Importance (Mean |SHAP|)",
) -> go.Figure:
    """Global feature importance chart using mean absolute SHAP values.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: Feature names list.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True).tail(20)

    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker=dict(
            color=importance_df["importance"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Mean |SHAP|"),
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f1f5f9")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mean |SHAP Value|",
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def cache_shap_values(shap_values: np.ndarray, path: str = config.SHAP_CACHE_PATH) -> None:
    """Persist SHAP values to disk to avoid recomputation.

    Args:
        shap_values: SHAP values array to cache.
        path: Output file path.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(shap_values, path)
        logger.info("SHAP values cached to '%s'", path)
    except Exception as exc:
        logger.error("Failed to cache SHAP values: %s", exc)


def load_shap_cache(path: str = config.SHAP_CACHE_PATH) -> Optional[np.ndarray]:
    """Load cached SHAP values from disk.

    Args:
        path: Path to the cached file.

    Returns:
        Cached SHAP values array, or None if not found.
    """
    if not os.path.exists(path):
        logger.info("No SHAP cache found at '%s'.", path)
        return None
    try:
        vals = joblib.load(path)
        logger.info("SHAP cache loaded from '%s'", path)
        return vals
    except Exception as exc:
        logger.warning("Failed to load SHAP cache: %s", exc)
        return None
