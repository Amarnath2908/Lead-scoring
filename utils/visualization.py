"""
utils/visualization.py — Plotly charts for Dashboard and History pages.
"""
import plotly.graph_objects as go
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def plot_score_distribution(scores, title="Lead Score Distribution"):
    """Histogram of lead scores coloured by band."""
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=20, marker_color="#6366f1", opacity=0.85,
    ))
    fig.update_layout(title=title, xaxis_title="Lead Score", yaxis_title="Count",
                      height=350, **_DARK)
    return fig


def plot_priority_pie(priorities, title="Priority Breakdown"):
    """Donut chart of priority band counts."""
    counts = priorities.value_counts()
    colors = [config.BAND_COLORS.get(b, "#94a3b8") for b in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.45, marker=dict(colors=colors),
    ))
    fig.update_layout(title=title, height=350, **_DARK)
    return fig


def plot_conversion_bar(history_df, title="Predicted Outcomes"):
    """Bar chart of predicted Convert vs Not Convert."""
    if "prediction" not in history_df.columns or history_df.empty:
        return go.Figure()
    counts = history_df["prediction"].value_counts().sort_index()
    labels = {0: "Won't Convert", 1: "Will Convert"}
    colors = {0: "#ef4444", 1: "#10b981"}
    fig = go.Figure(go.Bar(
        x=[labels.get(k, str(k)) for k in counts.index],
        y=counts.values,
        marker_color=[colors.get(k, "#6366f1") for k in counts.index],
    ))
    fig.update_layout(title=title, yaxis_title="Count", height=350, **_DARK)
    return fig


def plot_gauge(score, title="Lead Score"):
    """Gauge chart showing a single lead score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"color": "#f1f5f9"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#64748b"),
            bar=dict(color="#6366f1"),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                {"range": [0, 25],  "color": "#312e81"},
                {"range": [25, 50], "color": "#3730a3"},
                {"range": [50, 75], "color": "#4338ca"},
                {"range": [75, 90], "color": "#6366f1"},
                {"range": [90, 100],"color": "#818cf8"},
            ],
        ),
    ))
    fig.update_layout(height=280, **_DARK)
    return fig


def plot_leads_by_source(df, col="Lead Source", title="Leads by Source"):
    """Horizontal bar of lead counts per source (from raw data)."""
    if col not in df.columns:
        return go.Figure()
    counts = df[col].value_counts().head(10)
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker_color="#6366f1",
    ))
    fig.update_layout(title=title, xaxis_title="Count", height=350,
                      yaxis=dict(autorange="reversed"), **_DARK)
    return fig
