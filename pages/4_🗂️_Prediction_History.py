"""
pages/4_🗂️_Prediction_History.py — Table of past predictions with filters and stats.
"""
import os, sys, json
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import get_history, get_stats, clear_history
from utils.visualization import plot_score_distribution, plot_priority_pie

st.set_page_config(page_title="History | LeadScore AI", page_icon="🗂️", layout="wide")
st.markdown('<h1 class="gradient-text">🗂️ Prediction History</h1>', unsafe_allow_html=True)

history = get_history(limit=1000)

if history.empty:
    st.info("No predictions yet. Use **🎯 Predict Lead** to start scoring leads.")
    st.stop()

# ── Stats row ─────────────────────────────────────────────────────────────────
stats = get_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", stats["total"])
c2.metric("Avg Lead Score", stats["avg_score"])
c3.metric("Predicted Converts", stats["converts"])
c4.metric("Predicted Conv Rate", f"{stats['conversion_rate']}%")

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    band_filter = st.multiselect(
        "Filter by Priority Band", options=history["priority"].unique().tolist(),
        default=history["priority"].unique().tolist(),
    )
with col2:
    pred_filter = st.multiselect(
        "Filter by Prediction", options=["Convert", "Won't Convert"],
        default=["Convert", "Won't Convert"],
    )

pred_map = {"Convert": 1, "Won't Convert": 0}
pred_vals = [pred_map[p] for p in pred_filter]
filtered = history[
    history["priority"].isin(band_filter) & history["prediction"].isin(pred_vals)
]

st.markdown(f"**Showing {len(filtered)} of {len(history)} records**")

# ── Data table ────────────────────────────────────────────────────────────────
display_cols = ["id", "timestamp", "prediction", "probability", "lead_score", "priority", "recommendation"]
available = [c for c in display_cols if c in filtered.columns]
st.dataframe(
    filtered[available].style.format({
        "probability": "{:.1%}",
    }),
    use_container_width=True, hide_index=True, height=400,
)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if "lead_score" in filtered.columns:
        st.plotly_chart(plot_score_distribution(filtered["lead_score"]), use_container_width=True)
with col2:
    if "priority" in filtered.columns:
        st.plotly_chart(plot_priority_pie(filtered["priority"]), use_container_width=True)

# ── Clear history ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚠️ Danger Zone"):
    if st.button("🗑️ Clear All History", type="secondary"):
        deleted = clear_history()
        st.success(f"Deleted {deleted} records. Refresh the page.")
        st.rerun()
