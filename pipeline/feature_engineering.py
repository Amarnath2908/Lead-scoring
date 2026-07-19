"""
pipeline/feature_engineering.py
Create derived features from pre-contact data.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds a website_engagement_score column (weighted combo of visit metrics)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Engagement score = weighted sum of normalised visit metrics
        cols = {
            "TotalVisits":                  config.ENGAGEMENT_VISIT_WEIGHT,
            "Total Time Spent on Website":  config.ENGAGEMENT_TIME_WEIGHT,
            "Page Views Per Visit":         config.ENGAGEMENT_PV_WEIGHT,
        }
        score = pd.Series(0.0, index=X.index)
        for col, w in cols.items():
            if col in X.columns:
                raw = pd.to_numeric(X[col], errors="coerce").fillna(0)
                lo, hi = raw.min(), raw.max()
                norm = (raw - lo) / (hi - lo) if hi > lo else 0.0
                score += w * norm
        X["website_engagement_score"] = score
        return X
