"""
utils/preprocessing.py — Data cleaning, leakage audit, and sklearn Pipeline.

Design:
  - All transforms are wrapped in a single sklearn Pipeline + ColumnTransformer
    so the same artifact is used identically at training and inference time.
  - Leakage columns are dropped BEFORE fitting anything.
  - Outlier capping uses IQR with no row drops.
  - Imputation strategy is explicit and documented per feature type.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Transformers
# ---------------------------------------------------------------------------

class SelectReplacer(BaseEstimator, TransformerMixin):
    """Replace the literal string 'Select' with NaN across all columns.

    'Select' is a placeholder value in the raw CRM export meaning the
    user did not choose an option. It must be treated as missing, not as
    a valid category, to avoid biasing imputation and encoding.
    Also pads any missing columns present during training with NaN.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "SelectReplacer":
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Pad missing columns with NaN to satisfy ColumnTransformer requirements
        if hasattr(self, "columns_"):
            for col in self.columns_:
                if col not in X.columns:
                    X[col] = np.nan
                    
        mask = X == "Select"
        replaced = int(mask.sum().sum())
        if replaced > 0:
            logger.info("SelectReplacer: replaced %d 'Select' values with NaN", replaced)
        X[mask] = np.nan
        return X


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap outliers in specified columns using the IQR method.

    Rows are NEVER dropped — only values are clipped to
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    Works with both DataFrames and numpy arrays.

    Args:
        columns: List of numeric column names to cap.
        multiplier: IQR fence multiplier (default from config).
    """

    def __init__(
        self,
        columns: List[str] = None,
        multiplier: float = config.IQR_MULTIPLIER,
    ) -> None:
        self.columns = columns or config.OUTLIER_COLUMNS
        self.multiplier = multiplier
        self._bounds: dict = {}

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def fit(self, X, y=None) -> "IQRCapper":
        X = self._to_df(X)
        for col in self.columns:
            if col in X.columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self._bounds[col] = (
                    q1 - self.multiplier * iqr,
                    q3 + self.multiplier * iqr,
                )
                logger.debug(
                    "IQRCapper: %s bounds = [%.2f, %.2f]",
                    col,
                    *self._bounds[col],
                )
        return self

    def transform(self, X) -> pd.DataFrame:
        is_df = isinstance(X, pd.DataFrame)
        df = self._to_df(X).copy()
        for col, (lower, upper) in self._bounds.items():
            if col in df.columns:
                before = df[col].copy()
                df[col] = df[col].clip(lower=lower, upper=upper)
                capped = (before != df[col]).sum()
                if capped > 0:
                    logger.info("IQRCapper: capped %d values in '%s'", capped, col)
        return df if is_df else df.values


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set frequency.

    Used for high-cardinality columns to avoid dimensionality explosion
    with OneHotEncoder. Unknown categories at inference time get frequency 0.
    Works with both DataFrames (by column name) and numpy arrays (positionally).

    Args:
        columns: High-cardinality column names to encode.
    """

    def __init__(self, columns: List[str] = None) -> None:
        self.columns = columns or []
        self._freq_maps: dict = {}  # keyed by column name
        self._pos_maps: dict  = {}  # keyed by positional index (numpy fallback)

    def fit(self, X, y=None) -> "FrequencyEncoder":
        if isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns:
                    freq = X[col].value_counts(normalize=True)
                    self._freq_maps[col] = freq.to_dict()
        else:
            # numpy array: columns are positional
            df = pd.DataFrame(X)
            for i in range(df.shape[1]):
                freq = df[i].value_counts(normalize=True)
                self._pos_maps[i] = freq.to_dict()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col, freq_map in self._freq_maps.items():
                if col in X.columns:
                    X[col] = X[col].map(freq_map).fillna(0.0)
            return X
        else:
            # numpy array: use positional maps
            df = pd.DataFrame(X)
            maps = self._pos_maps if self._pos_maps else {
                i: self._freq_maps.get(col, {})
                for i, col in enumerate(self.columns)
            }
            for i, freq_map in maps.items():
                if i < df.shape[1]:
                    df[i] = df[i].map(freq_map).fillna(0.0)
            return df.values.astype(float)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, logging shape and dtypes summary.

    Args:
        path: Absolute path to Lead Scoring CSV.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place 'Lead Scoring.csv' in the data/ directory."
        )
    df = pd.read_csv(path)
    logger.info(
        "Loaded raw data: %d rows × %d cols from '%s'",
        len(df),
        df.shape[1],
        path,
    )
    return df


def audit_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Document and optionally drop post-contact leakage columns.

    Leakage columns are fields populated only AFTER a sales rep has
    qualified or worked a lead. Including them inflates model performance
    but renders the model useless at lead-intake time.

    If config.USE_LEAKAGE_COLS is False (default/production mode),
    the columns are dropped before any modeling step.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with leakage columns removed (or retained if
        USE_LEAKAGE_COLS is True).
    """
    present = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]
    if not present:
        logger.info("Leakage audit: none of the leakage columns found in dataset.")
        return df

    if config.USE_LEAKAGE_COLS:
        logger.warning(
            "USE_LEAKAGE_COLS=True — leakage columns RETAINED. "
            "This model is only valid for retrospective analysis, NOT live scoring."
        )
        return df

    df = df.drop(columns=present)
    logger.info(
        "Leakage audit: dropped %d post-contact columns: %s",
        len(present),
        present,
    )
    return df


def drop_ids_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID columns and duplicate rows, logging counts.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    # Drop IDs
    id_cols = [c for c in config.ID_COLUMNS if c in df.columns]
    df = df.drop(columns=id_cols)
    logger.info("Dropped %d ID columns: %s", len(id_cols), id_cols)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    logger.info("Dropped %d duplicate rows (kept %d)", dropped, len(df))
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = config.MISSING_THRESHOLD) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns whose missing rate exceeds the threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed missing fraction (from config).

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    missing_rate = df.isnull().mean()
    to_drop = missing_rate[missing_rate > threshold].index.tolist()
    df = df.drop(columns=to_drop)
    if to_drop:
        logger.info(
            "Dropped %d high-missing columns (>%.0f%%): %s",
            len(to_drop),
            threshold * 100,
            to_drop,
        )
    return df, to_drop


def identify_column_types(
    df: pd.DataFrame, target_col: str = "Converted"
) -> Tuple[List[str], List[str], List[str]]:
    """Identify numeric, low-cardinality categorical, and high-cardinality categorical columns.

    Args:
        df: DataFrame after cleaning (no target column needed here).
        target_col: Name of the target column (excluded from feature lists).

    Returns:
        Tuple of (numeric_cols, low_card_cat_cols, high_card_cat_cols).
    """
    feature_df = df.drop(columns=[target_col], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    low_card_cats = [
        c for c in cat_cols
        if feature_df[c].nunique() <= config.HIGH_CARDINALITY_THRESHOLD
    ]
    high_card_cats = [
        c for c in cat_cols
        if feature_df[c].nunique() > config.HIGH_CARDINALITY_THRESHOLD
    ]

    logger.info(
        "Column types — numeric: %d, low-card cat: %d, high-card cat: %d",
        len(numeric_cols),
        len(low_card_cats),
        len(high_card_cats),
    )
    logger.debug("Numeric: %s", numeric_cols)
    logger.debug("Low-card categorical: %s", low_card_cats)
    logger.debug("High-card categorical: %s", high_card_cats)

    return numeric_cols, low_card_cats, high_card_cats


def build_preprocessor(
    numeric_cols: List[str],
    low_card_cats: List[str],
    high_card_cats: List[str],
) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for preprocessing.

    Numeric pipeline:
        1. Median imputation (robust to skew/outliers).
        2. StandardScaler (z-score normalization after IQR capping).

    Low-cardinality categorical pipeline:
        1. Mode imputation (most frequent).
        2. OneHotEncoder (drop='first' to avoid multicollinearity).

    High-cardinality categorical pipeline:
        1. Constant 'Unknown' imputation (explicit missing flag,
           not silently folded into most-frequent).
        2. FrequencyEncoder (replace category with its training frequency).

    Note: IQR capping is applied BEFORE this ColumnTransformer
    (see build_full_pipeline).

    Args:
        numeric_cols: List of numeric feature names.
        low_card_cats: Low-cardinality categorical feature names.
        high_card_cats: High-cardinality categorical feature names.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )),
    ])

    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("freq_encoder", FrequencyEncoder(columns=high_card_cats)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if low_card_cats:
        transformers.append(("low_card_cat", low_card_pipeline, low_card_cats))
    if high_card_cats:
        transformers.append(("high_card_cat", high_card_pipeline, high_card_cats))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(
    numeric_cols: List[str],
    low_card_cats: List[str],
    high_card_cats: List[str],
) -> Pipeline:
    """Assemble the end-to-end preprocessing Pipeline.

    Steps:
        1. SelectReplacer — 'Select' → NaN
        2. IQRCapper     — cap outliers in known skewed columns
        3. ColumnTransformer — impute + encode + scale

    This single Pipeline object is saved as preprocessor.pkl and used
    identically at training and inference time.

    Args:
        numeric_cols: Numeric feature names.
        low_card_cats: Low-cardinality categorical feature names.
        high_card_cats: High-cardinality categorical feature names.

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor_ct = build_preprocessor(numeric_cols, low_card_cats, high_card_cats)

    pipeline = Pipeline([
        ("select_replacer", SelectReplacer()),
        ("iqr_capper", IQRCapper()),
        ("column_transformer", preprocessor_ct),
    ])

    logger.info("Built full preprocessing pipeline with %d steps", len(pipeline.steps))
    return pipeline


def get_feature_names_out(pipeline: Pipeline, numeric_cols, low_card_cats, high_card_cats) -> List[str]:
    """Extract output feature names from the fitted ColumnTransformer.

    Args:
        pipeline: Fitted full Pipeline.
        numeric_cols: Numeric column names (pre-transform).
        low_card_cats: Low-card categorical column names.
        high_card_cats: High-card categorical column names.

    Returns:
        List of feature names after transformation.
    """
    ct = pipeline.named_steps["column_transformer"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn
        names = list(numeric_cols)
        for col in low_card_cats:
            ohe = ct.named_transformers_["low_card_cat"].named_steps["encoder"]
            cats = ohe.categories_[low_card_cats.index(col)][1:]  # drop='first'
            names += [f"{col}_{c}" for c in cats]
        names += high_card_cats
        return names


def save_preprocessor(pipeline: Pipeline, path: str = config.PREPROCESSOR_PATH) -> None:
    """Persist the fitted pipeline to disk via joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Preprocessor saved to '%s'", path)


def load_preprocessor(path: str = config.PREPROCESSOR_PATH) -> Pipeline:
    """Load the fitted pipeline from disk.

    Args:
        path: Path to the saved .pkl file.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{path}'. Run the training script first."
        )
    pipeline = joblib.load(path)
    logger.info("Preprocessor loaded from '%s'", path)
    return pipeline
