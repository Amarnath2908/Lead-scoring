"""
pages/3_🎯_Predict_Lead.py — Form to enter one lead's details and get a prediction.
"""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.prediction import load_model, load_preprocessor, predict_single
from utils.scoring import score_lead
from utils.database import save_prediction
from utils.visualization import plot_gauge

st.set_page_config(page_title="Predict Lead | LeadScore AI", page_icon="🎯", layout="wide")
st.markdown('<h1 class="gradient-text">🎯 Predict Lead</h1>', unsafe_allow_html=True)

# ── Check model exists ────────────────────────────────────────────────────────
if not os.path.exists(config.MODEL_PATH):
    st.error("No trained model found. Run `python pipeline/train.py` first.")
    st.stop()


@st.cache_resource
def _load_artifacts():
    return load_model(), load_preprocessor()


model, preprocessor = _load_artifacts()

# ── Load dataset for dropdown values ──────────────────────────────────────────

@st.cache_data
def _load_raw():
    df = pd.read_csv(config.DATA_PATH)
    df.replace("Select", np.nan, inplace=True)
    return df

raw = _load_raw()

def _unique(col):
    """Return sorted unique non-null values for a column."""
    if col in raw.columns:
        vals = raw[col].dropna().unique().tolist()
        return sorted([str(v) for v in vals])
    return []


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("Fill in the lead attributes below.")

col1, col2, col3 = st.columns(3)
with col1:
    lead_origin = st.selectbox("Lead Origin", _unique("Lead Origin"))
    lead_source = st.selectbox("Lead Source", _unique("Lead Source"))
    last_activity = st.selectbox("Last Activity", _unique("Last Activity"))
    specialization = st.selectbox("Specialization", _unique("Specialization"))

with col2:
    total_visits = st.number_input("Total Visits", min_value=0, max_value=200, value=3)
    time_spent = st.slider("Total Time Spent on Website (min)", 0, 2500, 300)
    page_views = st.slider("Page Views Per Visit", 0.0, 20.0, 3.0, 0.5)
    what_matters = st.selectbox("What matters most in choosing a course",
                                _unique("What matters most to you in choosing a course"))

with col3:
    city = st.selectbox("City", _unique("City"))
    occupation = st.selectbox("Current Occupation", _unique("What is your current occupation"))
    do_not_email = st.radio("Do Not Email", ["No", "Yes"], horizontal=True)
    do_not_call = st.radio("Do Not Call", ["No", "Yes"], horizontal=True)
    free_copy = st.radio("Free copy of Mastering The Interview", ["No", "Yes"], horizontal=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict", type="primary", use_container_width=True):
    input_dict = {
        "Lead Origin": lead_origin,
        "Lead Source": lead_source,
        "Do Not Email": do_not_email,
        "Do Not Call": do_not_call,
        "TotalVisits": float(total_visits),
        "Total Time Spent on Website": float(time_spent),
        "Page Views Per Visit": float(page_views),
        "Last Activity": last_activity,
        "Specialization": specialization,
        "What matters most to you in choosing a course": what_matters,
        "City": city,
        "A free copy of Mastering The Interview": free_copy,
        "What is your current occupation": occupation,
    }

    try:
        input_df = pd.DataFrame([input_dict])
        with st.spinner("Running prediction..."):
            prediction, probability = predict_single(input_df, preprocessor, model)

        scoring = score_lead(probability)
        lead_score = scoring["lead_score"]
        priority = scoring["priority"]
        recommendation = scoring["recommendation"]
        band_color = scoring["band_color"]

        st.markdown("---")
        st.markdown("## 📊 Results")

        # ── Result cards ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            result_color = "#10b981" if prediction == 1 else "#ef4444"
            result_text = "✅ WILL CONVERT" if prediction == 1 else "❌ WON'T CONVERT"
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {result_color};">
              <div class="kpi-label">Prediction</div>
              <div class="kpi-value" style="color:{result_color}">{result_text}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {band_color};">
              <div class="kpi-label">Lead Score</div>
              <div class="kpi-value">{lead_score}</div>
              <div style="color:{band_color};font-weight:700;margin-top:0.25rem;">{priority}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {band_color};">
              <div class="kpi-label">Recommended Action</div>
              <div style="font-size:1.1rem;color:{band_color};font-weight:600;margin-top:0.5rem;">
                {recommendation}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Probability bar + gauge ───────────────────────────────────────────
        st.markdown(f"**Conversion Probability:** {probability:.1%}")
        st.progress(probability)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_gauge(lead_score), use_container_width=True)
        with c2:
            st.markdown("### 🏷️ Priority Summary")
            for band, (lo, hi) in config.SCORE_BANDS.items():
                color = config.BAND_COLORS[band]
                active = band == priority
                marker = "→ " if active else "　"
                w = "800" if active else "400"
                bg = "rgba(255,255,255,0.06)" if active else "transparent"
                bl = f"3px solid {color}" if active else "3px solid transparent"
                st.markdown(
                    f'<div style="padding:0.4rem 0.75rem;margin:0.2rem 0;border-radius:8px;'
                    f'background:{bg};border-left:{bl};">'
                    f'<span style="color:{color};font-weight:{w};">{marker}{band}</span>'
                    f'<span style="color:#64748b;font-size:0.8rem;float:right;">{lo}–{hi}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Save to DB ────────────────────────────────────────────────────────
        save_prediction(input_dict, prediction, probability, lead_score, priority, recommendation)
        st.caption("✅ Prediction saved to history.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        with st.expander("Details"):
            st.code(traceback.format_exc())
