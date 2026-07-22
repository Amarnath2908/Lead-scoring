"""
pages/2_📊_Dashboard.py — Charts: conversion rate, score distribution, leads by source, prediction stats.
"""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import get_history, get_stats
from utils.visualization import (
    plot_score_distribution, plot_priority_pie,
    plot_conversion_bar, plot_leads_by_source,
)

st.set_page_config(page_title="Dashboard | LeadScore AI", page_icon="📊", layout="wide")
st.markdown('<h1 class="gradient-text">📊 Dashboard</h1>', unsafe_allow_html=True)

# ── KPI row from prediction history 
stats = get_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", stats["total"])
c2.metric("Avg Lead Score", stats["avg_score"])
c3.metric("Predicted Converts", stats["converts"])
c4.metric("Predicted Conv Rate", f"{stats['conversion_rate']}%")

st.markdown("---")
# ── Dataset-level charts 
st.markdown("### 📈 Dataset Overview")
if os.path.exists(config.DATA_PATH):
    raw = pd.read_csv(config.DATA_PATH)
    raw.replace("Select", np.nan, inplace=True)

    col1, col2 = st.columns(2)
    with col1:
        if "Converted" in raw.columns:
            conv = raw["Converted"].value_counts()
            labels = {0: "Not Converted", 1: "Converted"}
            import plotly.graph_objects as go
            fig = go.Figure(go.Pie(
                labels=[labels.get(k, str(k)) for k in conv.index],
                values=conv.values, hole=0.45,  # donut chart
                marker=dict(colors=["#ef4444", "#10b981"]),
            ))
            fig.update_layout(title="Dataset Conversion Rate", height=350,
                              template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig_src = plot_leads_by_source(raw)
        st.plotly_chart(fig_src, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "Lead Origin" in raw.columns:
            fig_origin = plot_leads_by_source(raw, col="Lead Origin", title="Leads by Origin")
            st.plotly_chart(fig_origin, use_container_width=True)
else:
    st.info("Dataset not found — dataset charts will appear after placing `Lead Scoring.csv`.")

# ── Prediction history charts 
st.markdown("---")
st.markdown("### 📊 Prediction History Charts")
history = get_history(limit=1000)
if history.empty:
    st.info("No predictions yet. Use **🎯 Predict Lead** to score leads.")
else:
    col1, col2 = st.columns(2)
    with col1:
        if "lead_score" in history.columns:
            fig = plot_score_distribution(history["lead_score"])
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "priority" in history.columns:
            fig = plot_priority_pie(history["priority"])
            st.plotly_chart(fig, use_container_width=True)

    fig_bar = plot_conversion_bar(history)
    st.plotly_chart(fig_bar, use_container_width=True)

