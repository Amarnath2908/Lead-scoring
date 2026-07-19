"""
pages/1_🏠_Home.py — Landing page: what the app does, dataset summary.
"""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Home | LeadScore AI", page_icon="🏠", layout="wide")
st.markdown('<h1 class="gradient-text">🏠 Home</h1>', unsafe_allow_html=True)

# ── What the app does ─────────────────────────────────────────────────────────
st.markdown("""
### What is LeadScore AI?

**X Education** receives thousands of leads but only **~30 %** convert into paying customers.
This app uses machine learning to predict each lead's conversion probability and assigns a
**Lead Score (0–100)** so the sales team can focus on the highest-intent prospects first.
""")

st.markdown("---")

# ── Priority band legend ─────────────────────────────────────────────────────
st.markdown("### 🏷️ Priority Bands")
cols = st.columns(5)
for col, (band, (lo, hi)) in zip(cols, config.SCORE_BANDS.items()):
    color = config.BAND_COLORS[band]
    action = config.BAND_ACTIONS[band]
    with col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:1rem;text-align:center;border-top:3px solid {color};">
          <div style="font-weight:800;color:{color};font-size:1.1rem;">{band}</div>
          <div style="color:#94a3b8;font-size:0.8rem;margin:0.25rem 0;">{lo}–{hi}</div>
          <div style="color:#cbd5e1;font-size:0.75rem;">{action}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Dataset summary ───────────────────────────────────────────────────────────
st.markdown("### 📦 Dataset Summary")
if os.path.exists(config.DATA_PATH):
    df = pd.read_csv(config.DATA_PATH)
    df.replace("Select", np.nan, inplace=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads", f"{len(df):,}")
    c2.metric("Features", df.shape[1])
    conv_rate = round(df["Converted"].mean() * 100, 1) if "Converted" in df.columns else 0
    c3.metric("Conversion Rate", f"{conv_rate}%")
    c4.metric("Missing Cells", f"{int(df.isnull().sum().sum()):,}")

    with st.expander("📋 Sample rows"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Column dtypes"):
        dtypes = df.dtypes.reset_index()
        dtypes.columns = ["Column", "Type"]
        dtypes["Type"] = dtypes["Type"].astype(str)
        st.dataframe(dtypes, use_container_width=True, hide_index=True)
else:
    st.warning("Dataset not found. Place `Lead Scoring.csv` in the project root.")
