"""
tests/test_preprocessing.py — Unit tests for preprocessing utilities.
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (
    SelectReplacer, IQRCapper, FrequencyEncoder,
    drop_high_missing, build_full_pipeline,
)
import config


# ── Fixtures 
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "visits":   [1, 2, 3, 100, 5],
        "source":   ["Google", "Select", "Direct", "Select", "Organic"],
        "city":     ["Mumbai", "Delhi", "Mumbai", "Chennai", "Delhi"],
        "score":    [0.1, 0.5, np.nan, 0.8, 0.3],
        "Converted": [0, 1, 0, 1, 0],
    })


# ── SelectReplacer 
def test_select_replacer_converts_select_to_nan():
    df = pd.DataFrame({"a": ["Select", "Google", "Select"]})
    result = SelectReplacer().fit_transform(df)
    assert result["a"].isna().sum() == 2


def test_select_replacer_preserves_other_values():
    df = pd.DataFrame({"a": ["Google", "Direct", "Organic"]})
    result = SelectReplacer().fit_transform(df)
    assert result["a"].isna().sum() == 0
    assert list(result["a"]) == ["Google", "Direct", "Organic"]


def test_select_replacer_handles_mixed_types():
    df = pd.DataFrame({"a": ["Select", 1, None, "Select"]})
    result = SelectReplacer().fit_transform(df)
    assert result["a"].iloc[0] is np.nan or pd.isna(result["a"].iloc[0])


# ── IQRCapper 
def test_iqr_capper_clips_outliers():
    df = pd.DataFrame({"TotalVisits": [1, 2, 3, 4, 5, 1000]})
    capper = IQRCapper(columns=["TotalVisits"])
    capper.fit(df)
    result = capper.transform(df)
    assert result["TotalVisits"].max() < 1000


def test_iqr_capper_does_not_drop_rows():
    df = pd.DataFrame({"TotalVisits": [1, 2, 3, 4, 5, 1000]})
    capper = IQRCapper(columns=["TotalVisits"])
    capper.fit(df)
    result = capper.transform(df)
    assert len(result) == len(df)


def test_iqr_capper_preserves_non_outlier_values():
    df = pd.DataFrame({"TotalVisits": [1, 2, 3, 4, 5]})
    capper = IQRCapper(columns=["TotalVisits"])
    capper.fit(df)
    result = capper.transform(df)
    # Values within normal range should be unchanged
    assert result["TotalVisits"].iloc[2] == 3


def test_iqr_capper_ignores_missing_columns():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    capper = IQRCapper(columns=["TotalVisits"])
    capper.fit(df)
    result = capper.transform(df)
    assert "other_col" in result.columns


# ── FrequencyEncoder 
def test_frequency_encoder_maps_frequencies():
    df = pd.DataFrame({"city": ["Mumbai", "Mumbai", "Delhi", "Mumbai", "Delhi"]})
    enc = FrequencyEncoder(columns=["city"])
    enc.fit(df)
    result = enc.transform(df)
    # Mumbai appears 3/5 = 0.6, Delhi appears 2/5 = 0.4
    assert abs(result["city"].iloc[0] - 0.6) < 1e-6
    assert abs(result["city"].iloc[2] - 0.4) < 1e-6


def test_frequency_encoder_unknown_gets_zero():
    df_train = pd.DataFrame({"city": ["Mumbai", "Delhi"]})
    df_test  = pd.DataFrame({"city": ["Unknown City"]})
    enc = FrequencyEncoder(columns=["city"])
    enc.fit(df_train)
    result = enc.transform(df_test)
    assert result["city"].iloc[0] == 0.0


# ── drop_high_missing 
def test_drop_high_missing_respects_threshold():
    df = pd.DataFrame({
        "good_col":  [1, 2, 3, 4, 5],
        "bad_col":   [np.nan, np.nan, np.nan, np.nan, 1],   # 80% missing
        "ok_col":    [1, np.nan, 3, 4, 5],                  # 20% missing
    })
    result, dropped = drop_high_missing(df, threshold=0.40)
    assert "bad_col" in dropped
    assert "good_col" not in dropped
    assert "ok_col" not in dropped


def test_drop_high_missing_keeps_all_when_none_exceed():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result, dropped = drop_high_missing(df, threshold=0.40)
    assert len(dropped) == 0
    assert result.shape[1] == 2


# ── Full pipeline smoke test 
def test_pipeline_fit_transform_shape():
    numeric_cols = ["visits", "score"]
    low_card     = ["source"]
    high_card    = []
    df = pd.DataFrame({
        "visits": [1.0, 2.0, 3.0, 4.0, 5.0],
        "score":  [0.1, 0.2, 0.3, 0.4, 0.5],
        "source": ["Google", "Direct", "Organic", "Google", "Direct"],
    })
    pipeline = build_full_pipeline(numeric_cols, low_card, high_card)
    result = pipeline.fit_transform(df)
    # Should have rows preserved and columns >= original numeric
    assert result.shape[0] == len(df)
    assert result.shape[1] >= len(numeric_cols)
