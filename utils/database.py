"""
utils/database.py — SQLite persistence for prediction history with auto-recovery.
"""
import json, os, sqlite3, logging
from datetime import datetime
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _reset_db():
    """Remove corrupted DB file if it exists."""
    if os.path.exists(config.DB_PATH):
        try:
            os.remove(config.DB_PATH)
            logger.warning("Corrupted database removed. A new database will be created.")
        except Exception as e:
            logger.error(f"Failed to delete corrupted database: {e}")


def init_db():
    """Create the predictions table if it doesn't exist. Re-creates if corrupted."""
    os.makedirs(config.DB_DIR, exist_ok=True)
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT    NOT NULL,
                    input_json     TEXT    NOT NULL,
                    prediction     INTEGER NOT NULL,
                    probability    REAL    NOT NULL,
                    lead_score     INTEGER NOT NULL,
                    priority       TEXT    NOT NULL,
                    recommendation TEXT    NOT NULL
                )
            """)
            conn.commit()
    except sqlite3.DatabaseError as e:
        logger.error(f"Database error encountered: {e}. Resetting database...")
        _reset_db()
        # Retry creation with clean DB file
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT    NOT NULL,
                    input_json     TEXT    NOT NULL,
                    prediction     INTEGER NOT NULL,
                    probability    REAL    NOT NULL,
                    lead_score     INTEGER NOT NULL,
                    priority       TEXT    NOT NULL,
                    recommendation TEXT    NOT NULL
                )
            """)
            conn.commit()


def save_prediction(input_dict, prediction, probability, lead_score, priority, recommendation):
    """Insert one prediction row."""
    init_db()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, input_json, prediction, probability, lead_score, priority, recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(),
                 json.dumps(input_dict, default=str),
                 int(prediction), float(probability),
                 int(lead_score), str(priority), str(recommendation)),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        logger.error(f"Failed to save prediction due to DB error: {e}")
        _reset_db()
        init_db()


def get_history(limit=500):
    """Return recent predictions as a DataFrame."""
    init_db()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM predictions ORDER BY id DESC LIMIT {int(limit)}", conn
            )
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return pd.DataFrame()


def get_stats():
    """Aggregate stats from prediction history."""
    init_db()
    default = {"total": 0, "avg_score": 0, "converts": 0, "conversion_rate": 0}
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions")
            total = cur.fetchone()[0] or 0
            if total == 0:
                return default
            cur.execute("SELECT AVG(lead_score) FROM predictions")
            avg = round(cur.fetchone()[0] or 0, 1)
            cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
            converts = cur.fetchone()[0] or 0
        return {
            "total": total,
            "avg_score": avg,
            "converts": converts,
            "conversion_rate": round(converts / total * 100, 1),
        }
    except Exception as e:
        logger.error(f"Error computing stats: {e}")
        return default


def clear_history():
    """Delete all rows and return count deleted."""
    init_db()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            cur = conn.execute("DELETE FROM predictions")
            conn.commit()
            return cur.rowcount
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return 0
