"""
pipeline/model_training.py
Train 6 candidate models with probability calibration (CalibratedClassifierCV),
evaluate on accuracy (+ F1 for reference), return the best-accuracy model and all metrics.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_models():
    """Return dict of model_name → unfitted estimator."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=config.RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, algorithm="SAMME", random_state=config.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }


def train_all(X_train, y_train, X_test, y_test):
    """Train all candidates with probability calibration, return (best_model, best_name, all_metrics).

    Selection criterion: highest **accuracy** on the test set.
    Probability calibration via CalibratedClassifierCV ensures scores span full 0-100 range smoothly.
    """
    raw_models = get_models()
    results = []
    best_acc, best_model, best_name = -1, None, None

    for name, raw_model in raw_models.items():
        print(f"  Training & Calibrating {name} ...")
        calibrated_model = CalibratedClassifierCV(estimator=raw_model, method="sigmoid", cv=5)
        calibrated_model.fit(X_train, y_train)
        
        preds = calibrated_model.predict(X_test)
        probs = calibrated_model.predict_proba(X_test)[:, 1]
        
        acc = round(accuracy_score(y_test, preds), 4)
        f1 = round(f1_score(y_test, preds, zero_division=0), 4)
        min_score = round(float(probs.min()) * 100)
        max_score = round(float(probs.max()) * 100)
        
        print(f"    {name}: accuracy={acc}  f1={f1}  (score range: {min_score} to {max_score})")
        results.append({
            "model_name": name,
            "accuracy": acc,
            "f1": f1,
            "min_score": min_score,
            "max_score": max_score,
        })

        if acc > best_acc:
            best_acc = acc
            best_model = calibrated_model
            best_name = name

    return best_model, best_name, results
