"""
pipeline/data_preprocessing.py
Load raw CSV, clean nulls, drop leakage/ID columns, fix dtypes.
Returns a reusable sklearn ColumnTransformer pipeline step.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------

class SelectReplacer(BaseEstimator, TransformerMixin):
    """Replace the CRM placeholder 'Select' with NaN, and pad any
    columns that were present during training but missing at inference."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        # Pad missing columns
        for col in self.columns_:
            if col not in X.columns:
                X[col] = np.nan
        X[X == "Select"] = np.nan
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set frequency.
    Works with both DataFrames and numpy arrays (inside ColumnTransformer)."""

    def __init__(self, columns=None):
        self.columns = columns or []
        self._freq_maps = {}
        self._pos_maps = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns:
                    self._freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        else:
            df = pd.DataFrame(X)
            for i in range(df.shape[1]):
                self._pos_maps[i] = df[i].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col, fm in self._freq_maps.items():
                if col in X.columns:
                    X[col] = X[col].map(fm).fillna(0.0)
            return X
        else:
            df = pd.DataFrame(X)
            maps = self._pos_maps if self._pos_maps else {
                i: self._freq_maps.get(c, {}) for i, c in enumerate(self.columns)
            }
            for i, fm in maps.items():
                if i < df.shape[1]:
                    df[i] = df[i].map(fm).fillna(0.0)
            return df.values.astype(float)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def load_and_clean(path=config.DATA_PATH):
    """Load raw CSV, drop leakage/ID cols, duplicates, 'Select'→NaN."""
    df = pd.read_csv(path)
    # Drop leakage
    leakage = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]
    df.drop(columns=leakage, inplace=True)
    # Drop IDs
    ids = [c for c in config.ID_COLUMNS if c in df.columns]
    df.drop(columns=ids, inplace=True)
    # Duplicates
    df.drop_duplicates(inplace=True)
    # 'Select' → NaN
    df.replace("Select", np.nan, inplace=True)
    # Drop high-missing columns
    missing = df.isnull().mean()
    drop_cols = missing[missing > config.MISSING_THRESHOLD].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    return df


def classify_columns(df, target="Converted"):
    """Split feature columns into numeric / low-card cat / high-card cat."""
    features = df.drop(columns=[target], errors="ignore")
    num = features.select_dtypes(include=[np.number]).columns.tolist()
    cats = features.select_dtypes(include=["object", "category"]).columns.tolist()
    low = [c for c in cats if features[c].nunique() <= config.HIGH_CARDINALITY_THRESHOLD]
    high = [c for c in cats if features[c].nunique() > config.HIGH_CARDINALITY_THRESHOLD]
    return num, low, high


def build_preprocessor(num_cols, low_cats, high_cats):
    """Build the ColumnTransformer with impute + encode + scale."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("freq", FrequencyEncoder(columns=high_cats)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_cats:
        transformers.append(("low", low_pipe, low_cats))
    if high_cats:
        transformers.append(("high", high_pipe, high_cats))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(num_cols, low_cats, high_cats):
    """SelectReplacer → ColumnTransformer (single Pipeline object)."""
    ct = build_preprocessor(num_cols, low_cats, high_cats)
    return Pipeline([
        ("select_replacer", SelectReplacer()),
        ("column_transformer", ct),
    ])
