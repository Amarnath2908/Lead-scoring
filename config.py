"""
config.py — Central configuration for LeadScore AI.
Every constant, path, and threshold lives here.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "Lead Scoring.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DB_DIR = os.path.join(PROJECT_ROOT, "database")
DB_PATH = os.path.join(DB_DIR, "predictions.db")

# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
MISSING_THRESHOLD = 0.40          # drop columns with >40 % missing
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns that leak future information (filled only after sales contact)
LEAKAGE_COLUMNS = [
    "Tags", "Lead Quality", "Last Notable Activity",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score",
    "Asymmetrique Activity Index", "Asymmetrique Profile Index",
]

# ID columns — no predictive value
ID_COLUMNS = ["Prospect ID", "Lead Number"]

# Numeric columns to IQR-cap
OUTLIER_COLUMNS = ["TotalVisits", "Page Views Per Visit"]
IQR_MULTIPLIER = 1.5

# Cardinality threshold: ≤ this → OneHot, > this → frequency-encode
HIGH_CARDINALITY_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Lead-score bands
# ---------------------------------------------------------------------------
SCORE_BANDS = {
    "Very Hot":  (90, 100),
    "Hot":       (75, 89),
    "Warm":      (50, 74),
    "Cold":      (25, 49),
    "Very Cold": (0, 24),
}

BAND_ACTIONS = {
    "Very Hot":  "Call immediately",
    "Hot":       "Same-day call",
    "Warm":      "Follow-up email within 24 h",
    "Cold":      "Add to nurture campaign",
    "Very Cold": "Deprioritize",
}

BAND_COLORS = {
    "Very Hot":  "#ef4444",
    "Hot":       "#f97316",
    "Warm":      "#eab308",
    "Cold":      "#3b82f6",
    "Very Cold": "#6366f1",
}

# Feature-engineering weights
ENGAGEMENT_VISIT_WEIGHT = 0.3
ENGAGEMENT_TIME_WEIGHT = 0.5
ENGAGEMENT_PV_WEIGHT = 0.2
