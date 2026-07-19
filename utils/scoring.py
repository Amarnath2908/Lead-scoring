"""
utils/scoring.py — Lead Score computation and priority classification.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_lead_score(probability: float) -> int:
    """Convert probability [0, 1] → Lead Score [0, 100]."""
    return max(0, min(100, round(float(probability) * 100)))


def get_priority_band(score: int) -> str:
    """Map a Lead Score to its named priority band."""
    for band, (lo, hi) in config.SCORE_BANDS.items():
        if lo <= score <= hi:
            return band
    return "Very Cold"


def get_recommendation(band: str) -> str:
    """Return a sales action string for the given band."""
    return config.BAND_ACTIONS.get(band, "Review manually")


def get_band_color(band: str) -> str:
    """Return hex colour for the given band."""
    return config.BAND_COLORS.get(band, "#94a3b8")


def score_lead(probability: float) -> dict:
    """One-call helper: probability → full scoring dict."""
    score = compute_lead_score(probability)
    band = get_priority_band(score)
    return {
        "lead_score": score,
        "priority": band,
        "recommendation": get_recommendation(band),
        "band_color": get_band_color(band),
    }
