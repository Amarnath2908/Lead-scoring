"""
app.py — Streamlit entry point for LeadScore AI.
"""
import os, sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.database import init_db

st.set_page_config(
    page_title="LeadScore AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise DB on first load
init_db()

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
}
.gradient-text {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
}
.kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; margin-top: 0.25rem; }
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.band-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0.75rem; border-radius: 8px; margin: 0.25rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Landing content ───────────────────────────────────────────────────────────
st.markdown('<h1 class="gradient-text">🎯 LeadScore AI</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Lead Scoring & Conversion Prediction** for X Education")

# Model status
model_ready = os.path.exists(config.MODEL_PATH)
if model_ready:
    st.success("✅ Model loaded and ready — navigate to **Predict Lead** to start scoring.")
else:
    st.warning("⚠️ No trained model found. Run `python pipeline/train.py` to train.")

st.markdown("---")
st.markdown("### How it works")
st.markdown("""
1. **Enter lead details** on the Predict Lead page.
2. The ML model predicts conversion probability.
3. Probability is converted to a **Lead Score (0–100)**.
4. Each lead is assigned a **Priority Band** with a **recommended action**.
""")

st.markdown("### 🏷️ Priority Bands")
for band, (lo, hi) in config.SCORE_BANDS.items():
    color = config.BAND_COLORS[band]
    action = config.BAND_ACTIONS[band]
    st.markdown(
        f'<div class="band-row" style="border-left:4px solid {color};">'
        f'<span style="color:{color};font-weight:700;">{band}</span>'
        f'<span style="color:#94a3b8;font-size:0.85rem;">{lo}–{hi} → {action}</span></div>',
        unsafe_allow_html=True,
    )
