"""
tests/test_scoring.py — Unit tests for scoring utilities.
"""
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.scoring import (
    compute_lead_score, get_priority_band,
    get_recommendation, score_dataframe, score_single,
)
import config


def test_compute_lead_score_zero_probability():
    assert compute_lead_score(0.0) == 0


def test_compute_lead_score_full_probability():
    assert compute_lead_score(1.0) == 100


def test_compute_lead_score_half_probability():
    assert compute_lead_score(0.5) == 50


def test_compute_lead_score_clamps_above_100():
    assert compute_lead_score(1.5) == 100


def test_compute_lead_score_clamps_below_zero():
    assert compute_lead_score(-0.1) == 0


def test_compute_lead_score_rounding():
    # 0.936 → round(93.6) = 94
    assert compute_lead_score(0.936) == 94


def test_get_priority_band_very_hot():
    assert get_priority_band(95) == "Very Hot"
    assert get_priority_band(90) == "Very Hot"
    assert get_priority_band(100) == "Very Hot"


def test_get_priority_band_hot():
    assert get_priority_band(75) == "Hot"
    assert get_priority_band(85) == "Hot"
    assert get_priority_band(89) == "Hot"


def test_get_priority_band_warm():
    assert get_priority_band(50) == "Warm"
    assert get_priority_band(62) == "Warm"
    assert get_priority_band(74) == "Warm"


def test_get_priority_band_cold():
    assert get_priority_band(25) == "Cold"
    assert get_priority_band(40) == "Cold"
    assert get_priority_band(49) == "Cold"


def test_get_priority_band_very_cold():
    assert get_priority_band(0) == "Very Cold"
    assert get_priority_band(12) == "Very Cold"
    assert get_priority_band(24) == "Very Cold"


def test_get_recommendation_returns_string():
    for band in config.SCORE_BANDS:
        rec = get_recommendation(band)
        assert isinstance(rec, str)
        assert len(rec) > 0


def test_get_recommendation_unknown_band():
    rec = get_recommendation("Unknown Band")
    assert isinstance(rec, str)


def test_score_single_returns_four_values():
    score, band, rec, color = score_single(0.92)
    assert 0 <= score <= 100
    assert band in config.SCORE_BANDS
    assert isinstance(rec, str)
    assert color.startswith("#")


def test_score_dataframe_adds_columns():
    df = pd.DataFrame({"probability": [0.95, 0.30, 0.60, 0.10, 0.80]})
    result = score_dataframe(df)
    assert "lead_score"     in result.columns
    assert "priority"       in result.columns
    assert "recommendation" in result.columns
    assert "band_color"     in result.columns


def test_score_dataframe_scores_in_range():
    df = pd.DataFrame({"probability": [0.0, 0.25, 0.5, 0.75, 1.0]})
    result = score_dataframe(df)
    assert result["lead_score"].between(0, 100).all()


def test_score_dataframe_raises_on_missing_column():
    df = pd.DataFrame({"other_col": [0.5]})
    with pytest.raises(KeyError):
        score_dataframe(df, proba_col="probability")


def test_score_dataframe_priority_values_valid():
    df = pd.DataFrame({"probability": [0.95, 0.80, 0.60, 0.35, 0.10]})
    result = score_dataframe(df)
    valid_bands = set(config.SCORE_BANDS.keys())
    assert set(result["priority"].unique()).issubset(valid_bands)
