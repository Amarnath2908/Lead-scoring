"""
utils/model_training.py — Train, compare, and persist 6 classifiers.

Run this script offline (before launching the Streamlit app) to produce:
    models/model.pkl
    models/preprocessor.pkl
    models/model_metadata.json

Primary selection metric: F1-Score (see config.MODEL_SELECTION_METRIC).
Rationale: Missing a convertible lead (false negative) is costlier than
contacting a non-converter (false positive). F1 balances both without
silently optimising Accuracy, which is misleading under class imbalance.
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.feature_engineering import FeatureEngineer
from utils.preprocessing import (
    audit_leakage, build_full_pipeline, drop_high_missing,
    drop_ids_and_duplicates, identify_column_types, load_raw_data,
    save_preprocessor,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COL = "Converted"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(class_weight: Dict[int, float]) -> Dict[str, object]:
    """Return a dict of unfitted classifiers.

    Class imbalance is addressed via class_weight for sklearn estimators
    and scale_pos_weight for XGBoost.  SMOTE is intentionally avoided:
    synthetic oversampling at training time creates out-of-distribution
    examples that degrade real-world inference reliability.

    Args:
        class_weight: Dict mapping {0: w0, 1: w1} derived from training set.

    Returns:
        Mapping of model_name → unfitted classifier.
    """
    neg_weight = class_weight[0]
    pos_weight = class_weight[1]
    scale_pos = neg_weight / pos_weight  # for XGBoost

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=config.RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
            algorithm="SAMME",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
                             list, list, list]:
    """Load, clean, engineer features, split, and fit preprocessor.

    Returns:
        Tuple of (X_train_t, y_train, X_test_t, y_test,
                  numeric_cols, low_card_cats, high_card_cats).
        X_train_t and X_test_t are already transformed numpy arrays.
    """
    # --- Load ---
    df = load_raw_data()

    # --- Leakage audit (drops post-contact columns in production mode) ---
    df = audit_leakage(df)

    # --- Drop IDs + duplicates ---
    df = drop_ids_and_duplicates(df)

    # --- Drop high-missing columns ---
    df, _ = drop_high_missing(df)

    # --- Log class imbalance ---
    y_all = df[TARGET_COL]
    neg_count = (y_all == 0).sum()
    pos_count = (y_all == 1).sum()
    imbalance_ratio = neg_count / pos_count
    logger.info(
        "Class distribution — Converted=0: %d, Converted=1: %d (ratio %.2f:1)",
        neg_count, pos_count, imbalance_ratio,
    )

    # --- Feature engineering (before train/test split to learn bin edges) ---
    fe = FeatureEngineer()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # --- Stratified split BEFORE fitting to prevent data leakage ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train: %d rows | Test: %d rows | Stratified split",
        len(X_train_raw), len(X_test_raw),
    )

    # --- Fit feature engineer on TRAINING data only ---
    X_train_fe = fe.fit_transform(X_train_raw)
    X_test_fe = fe.transform(X_test_raw)

    # --- Identify column types AFTER feature engineering ---
    numeric_cols, low_card_cats, high_card_cats = identify_column_types(
        pd.concat([X_train_fe, pd.Series(y_train, name=TARGET_COL)], axis=1)
    )

    # --- Build and fit preprocessing pipeline on TRAINING data only ---
    preprocessor = build_full_pipeline(numeric_cols, low_card_cats, high_card_cats)
    X_train_t = preprocessor.fit_transform(X_train_fe)
    X_test_t = preprocessor.transform(X_test_fe)

    logger.info("Preprocessed shape — Train: %s | Test: %s", X_train_t.shape, X_test_t.shape)

    # --- Save pipeline (feature engineer + preprocessor bundled together) ---
    full_pipeline = Pipeline([
        ("feature_engineer", fe),
        ("preprocessor", preprocessor),
    ])
    save_preprocessor(full_pipeline, config.PREPROCESSOR_PATH)

    # Compute class weights for model init
    total = neg_count + pos_count
    class_weight = {0: total / (2 * neg_count), 1: total / (2 * pos_count)}

    return X_train_t, y_train.values, X_test_t, y_test.values, \
           numeric_cols, low_card_cats, high_card_cats, class_weight


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict:
    """Evaluate a trained model on held-out test set and via CV.

    Args:
        model: Fitted classifier.
        X_train: Training features (transformed).
        y_train: Training labels.
        X_test: Test features (transformed).
        y_test: Test labels.
        model_name: Human-readable model name.

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, cv_f1_mean, cv_f1_std.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=["f1", "roc_auc"],
        n_jobs=-1,
    )

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "model_name": model_name,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
        "cv_f1_mean": round(cv_results["test_f1"].mean(), 4),
        "cv_f1_std":  round(cv_results["test_f1"].std(), 4),
        "cv_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(
        "[%s] Acc=%.3f | Pre=%.3f | Rec=%.3f | F1=%.3f | AUC=%.3f | CV_F1=%.3f±%.3f",
        model_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        metrics["roc_auc"], metrics["cv_f1_mean"], metrics["cv_f1_std"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_and_select() -> None:
    """Full training pipeline: train all models, compare, save the best.

    Saves:
        models/model.pkl          — best fitted classifier
        models/preprocessor.pkl   — fitted feature + preprocessing pipeline
        models/model_metadata.json — metadata about the winning model
    """
    logger.info("=== Lead Scoring Model Training Started ===")
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test, \
    numeric_cols, low_card_cats, high_card_cats, class_weight = prepare_data()

    models = get_models(class_weight)
    all_metrics = []
    trained_models = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            all_metrics.append(metrics)
            trained_models[name] = model
        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)

    # --- Select best model by primary metric ---
    metric_key = config.MODEL_SELECTION_METRIC  # "f1"
    best_metrics = max(all_metrics, key=lambda m: m[metric_key])
    best_name = best_metrics["model_name"]
    best_model = trained_models[best_name]

    logger.info(
        "Best model: %s  (F1=%.4f, AUC=%.4f)",
        best_name, best_metrics["f1"], best_metrics["roc_auc"],
    )

    # --- Save best model ---
    joblib.dump(best_model, config.MODEL_PATH)
    logger.info("Model saved to '%s'", config.MODEL_PATH)

    # --- Save metadata ---
    metadata = {
        "best_model": best_name,
        "selection_metric": metric_key,
        "training_date": datetime.now().isoformat(),
        "best_metrics": best_metrics,
        "all_models_metrics": all_metrics,
        "config": {
            "test_size": config.TEST_SIZE,
            "cv_folds": config.CV_FOLDS,
            "random_state": config.RANDOM_STATE,
            "missing_threshold": config.MISSING_THRESHOLD,
            "use_leakage_cols": config.USE_LEAKAGE_COLS,
        },
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to '%s'", config.METADATA_PATH)
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    train_and_select()
