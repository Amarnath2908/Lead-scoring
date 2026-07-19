"""
utils/feature_engineering.py — Business-meaningful feature creation.

Every engineered feature is fully documented with its formula so it is
auditable and reproducible. All features are derived ONLY from pre-contact
fields to avoid data leakage.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that adds all engineered features.

    This transformer is inserted BEFORE the preprocessing pipeline so
    that the engineered columns flow through the same imputation / scaling
    steps as raw features.

    Features created:
        1. website_engagement_score  — weighted sum of visit metrics
        2. engagement_level          — Low / Medium / High bin of score
        3. interaction_score         — count of confirmed channel touchpoints
        4. visit_per_pageview        — ratio feature (efficiency proxy)
        5. time_per_visit            — average session depth proxy
    """

    def __init__(self) -> None:
        self._engagement_bins: list = []   # Learned bin edges from training data
        self._engagement_labels = config.ENGAGEMENT_BIN_LABELS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_website_engagement_score(df: pd.DataFrame) -> pd.Series:
        """Compute Website Engagement Score.

        Formula:
            WES = w_visit * norm(TotalVisits)
                + w_time  * norm(Total Time Spent on Website)
                + w_pv    * norm(Page Views Per Visit)

        where norm(x) = (x - x.min()) / (x.max() - x.min())
        applied column-wise; weights are defined in config.py.

        All three raw columns use pre-contact data (no leakage).

        Args:
            df: DataFrame containing the three source columns.

        Returns:
            Series with website engagement score in [0, 1].
        """
        cols = {
            "TotalVisits": config.ENGAGEMENT_VISIT_WEIGHT,
            "Total Time Spent on Website": config.ENGAGEMENT_TIME_WEIGHT,
            "Page Views Per Visit": config.ENGAGEMENT_PV_WEIGHT,
        }
        score = pd.Series(0.0, index=df.index)
        for col, weight in cols.items():
            if col in df.columns:
                raw = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                col_min, col_max = raw.min(), raw.max()
                if col_max > col_min:
                    norm = (raw - col_min) / (col_max - col_min)
                else:
                    norm = pd.Series(0.0, index=df.index)
                score += weight * norm
            else:
                logger.warning("Column '%s' not found for engagement score.", col)
        return score

    @staticmethod
    def _compute_interaction_score(df: pd.DataFrame) -> pd.Series:
        """Compute Interaction Score.

        Formula:
            IS = sum of binary channel activity flags:
                (Do Not Email == 'No') +
                (Do Not Call  == 'No') +
                (Search       == 'Yes') +
                (Magazine     == 'Yes') +
                (Newspaper Article == 'Yes') +
                (X Education Forums == 'Yes') +
                (Newspaper    == 'Yes') +
                (Digital Advertisement == 'Yes') +
                (Through Recommendations == 'Yes') +
                (Receive More Updates About Our Courses == 'Yes') +
                (Update me on Supply Chain Content == 'Yes')

        Higher scores indicate leads engaged across multiple channels.
        All fields are pre-contact questionnaire / tracking data.

        Args:
            df: Raw/cleaned DataFrame.

        Returns:
            Series with integer interaction count.
        """
        positive_yes = [
            "Search",
            "Magazine",
            "Newspaper Article",
            "X Education Forums",
            "Newspaper",
            "Digital Advertisement",
            "Through Recommendations",
            "Receive More Updates About Our Courses",
            "Update me on Supply Chain Content",
            "I agree to pay the amount through cheque",
        ]
        negative_no = ["Do Not Email", "Do Not Call"]

        score = pd.Series(0, index=df.index)
        for col in positive_yes:
            if col in df.columns:
                score += (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)
        for col in negative_no:
            if col in df.columns:
                score += (df[col].astype(str).str.strip().str.lower() == "no").astype(int)
        return score

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn bin edges for engagement_level from training data.

        Args:
            X: Training DataFrame.
            y: Ignored (required by sklearn API).

        Returns:
            Self.
        """
        eng_score = self._compute_website_engagement_score(X)
        # Use quantile-based bins so each bin has roughly equal population
        self._engagement_bins = [
            eng_score.quantile(0.0),
            eng_score.quantile(1 / config.ENGAGEMENT_N_BINS),
            eng_score.quantile(2 / config.ENGAGEMENT_N_BINS),
            eng_score.quantile(1.0) + 1e-9,  # Ensure max value is included
        ]
        # Deduplicate edges (can happen when data is sparse)
        self._engagement_bins = sorted(set(self._engagement_bins))
        logger.info("FeatureEngineer fitted. Engagement bins: %s", self._engagement_bins)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the DataFrame.

        Args:
            X: Input DataFrame (raw or partially cleaned).

        Returns:
            DataFrame with additional columns appended.
        """
        X = X.copy()

        # 1. Website Engagement Score (continuous, [0, 1])
        X["website_engagement_score"] = self._compute_website_engagement_score(X)

        # 2. Engagement Level (categorical bin of engagement score)
        if len(self._engagement_bins) >= 3:
            labels = self._engagement_labels[:len(self._engagement_bins) - 1]
            X["engagement_level"] = pd.cut(
                X["website_engagement_score"],
                bins=self._engagement_bins,
                labels=labels,
                include_lowest=True,
            ).astype(str)
        else:
            X["engagement_level"] = "Medium"

        # 3. Interaction Score (integer count of engaged channels)
        X["interaction_score"] = self._compute_interaction_score(X)

        # 4. Visit Per Pageview — ratio proxy for browsing efficiency.
        #    Formula: TotalVisits / max(Page Views Per Visit, 1)
        #    Guards against divide-by-zero with max(., 1).
        if "TotalVisits" in X.columns and "Page Views Per Visit" in X.columns:
            visits = pd.to_numeric(X["TotalVisits"], errors="coerce").fillna(0.0)
            pv = pd.to_numeric(X["Page Views Per Visit"], errors="coerce").fillna(1.0).clip(lower=1)
            X["visit_per_pageview"] = (visits / pv).round(4)

        # 5. Time Per Visit — average session depth proxy.
        #    Formula: Total Time Spent on Website / max(TotalVisits, 1)
        if "Total Time Spent on Website" in X.columns and "TotalVisits" in X.columns:
            time = pd.to_numeric(X["Total Time Spent on Website"], errors="coerce").fillna(0.0)
            v = pd.to_numeric(X["TotalVisits"], errors="coerce").fillna(1.0).clip(lower=1)
            X["time_per_visit"] = (time / v).round(4)

        logger.info(
            "FeatureEngineer: added 5 engineered features. New shape: %s", X.shape
        )
        return X
