# ui/streamlit_app.py
"""
GenAI Cement — Streamlit UI (Plotly visuals + compact CSS theme)

Drop this file into ui/ and restart your Streamlit app.
Requires src/*.py modules present in the repo root /src folder.
"""

import sys
import os
from pathlib import Path
import time
import joblib

# -------------------------------------------------------------------
# Ensure repo root is on sys.path so sibling modules in src/ can be imported.
# This makes the app robust to different working directories on Streamlit Cloud.
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project module imports (assumes src/*.py exist and use top-level imports)
from src.data_generator import gen_raw_grinding, gen_clinker, gen_quality, gen_altfuel, gen_cross
from src.trainers import (
    train_raw_grinding,
    train_clinker,
    train_quality,
    train_altfuel,
    train_cross,
    upload_and_register,
)
from src.ingest_simulator import write_all as ingest_write_all
from src.gcp_utils import write_config_to_bq, read_config_from_bq

# Third-party libs
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="GenAI Cement — Operator Dashboard",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Compact CSS — tighter spacing, card-like panels, accent color
# -------------------------------------------------------------------
COMPACT_CSS = """
<style>
/* Page background & font adjustments */
body {
    background-color: #f7fafc;
}

/* Accent color for headings & highlights */
.css-1d391kg { color: #0b6efd; } /* Streamlit internal class can vary; used for some heading accents */

/* Tighter spacing for Streamlit elements */
.css-1v3fvcr { padding-top: 4px; padding-bottom: 4px; } 
.reportview-container .main .block-container{
    padding-top: 8px;
    padding-right: 18px;
    padding-left: 18px;
    padding-bottom: 8px;
}

/* Card-like panels for sections */
.panel-card {
    background: white;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 1px 6px rgba(16,24,40,0.06);
    margin-bottom: 12px;
}

/* Metric tweaks */
.stMetricLabel, .stMetricValue {
    line-height: 1.1;
}

/* Smaller captions */
.small-caption { font-size: 12px; color: #6b7280; }

/* Make plotly charts have less vertical padding inside container */
[data-testid="stPlotlyChart"] > div {
    padding: 4px 0px;
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_csv(name):
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

# -------------------------------------------------------------------
# Plotly helpers (no statsmodels / trendline dependency)
# -------------------------------------------------------------------
def choose_chart_type(column_name):
    n = column_name.lower()
    if any(k in n for k in ["temp", "kiln", "feed", "rate", "blaine", "blain", "calorific"]):
        return "line"
    if any(k in n for k in ["variability", "efficiency", "predicted", "energy", "co2", "tsr"]):
        return "area"
    if any(k in n for k in ["count", "events", "off_spec", "alerts"]):
        return "bar"
    return "line"

def draw_plotly_timechart(df, x_col, y_col, title=None, height=320):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        st.info("No data for chart.")
        return
    chart_type = choose_chart_type(y_col)
    if chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title)
        fig.update_traces(mode="lines+markers", marker=dict(size=6))
    elif chart_type == "area":
        fig = px.area(df, x=x_col, y=y_col, title=title)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    else:
        fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(height=height, margin=dict(l=12, r=12, t=36, b=12), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def mini_sparkline(df, y_col, height=70):
    if df is None or y_col not in df.columns:
        return
    vals = df[y_col].tail(30).values
    fig = go.Figure(data=[go.Scatter(y=vals, mode="lines", line=dict(width=2))])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def gauge_metric(value, label, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        # default range around value
        min_val = 0 if value >= 0 else value*1.2
        max_val = value * 1.4 if value != 0 else 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        title={'text': label},
        gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': "#0b6efd"}},
    ))
    fig.update_layout(height=200, margin=dict(l=8, r=8, t=24, b=8), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def radar_chart(values, labels, title="Snapshot"):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name=title))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=360, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# UI small components
# -------------------------------------------------------------------
def show_kpi_cards(kpis):
    cols = st.columns(len(kpis))
    for c, (label, value, delta, unit) in zip(cols, kpis):
        with c:
            if delta is not None:
                st.metric(label, f"{value}{unit}", delta=delta)
            else:
                st.metric(label, f"{value}{unit}")

def show_ai_recommendation(text, details=None):
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("### 🔮 AI Recommendations")
    st.success(text)
    if details:
        st.caption(details)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Overview", "Raw & Grinding", "Clinker", "Quality", "Alt Fuel", "Cross", "Config", "Train"],
)

st.sidebar.markdown("---")
st.sidebar.write("Developer controls")
if st.sidebar.button("Regenerate synthetic CSVs (ingest)"):
    ingest_write_all()
    st.sidebar.success("Regenerated synthetic CSVs in `data/`")

# -------------------------------------------------------------------
# Overview page
# -------------------------------------------------------------------
if menu == "Overview":
    st.title("GenAI Cement — Operator Dashboard")
    # Load data
    df_rg = load_csv("raw_grinding.csv")
    df_cl = load_csv("clinker.csv")
    df_q = load_csv("quality.csv")
    df_af = load_csv("altfuel.csv")
    df_cross = load_csv("cross.csv")

    # Calculate KPIs (safe guards)
    energy_intensity = round(df_cross["predicted_energy"].mean(), 2) if (df_cross is not None and "predicted_energy" in df_cross.columns) else 9.58
    tsr = round(df_af["tsr"].mean(), 1) if (df_af is not None and "tsr" in df_af.columns) else 32
    off_spec = int(df_q.shape[0] * 0.02) if df_q is not None else 2
    co2 = round((df_cross["co2"].mean() if (df_cross is not None and "co2" in df_cross.columns) else 0.82), 2)

    kpis = [
        ("Energy Intensity", energy_intensity, "-0.4", " kWh/t"),
        ("Thermal Substitution Rate", tsr, None, " %"),
        ("Off-spec Events", off_spec, None, " /mo"),
        ("CO₂ Emissions", co2, "-0.1", " t/ton"),
    ]
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    show_kpi_cards(kpis)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([2,1])

    with left:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Raw Materials & Grinding")
        if df_rg is not None:
            # Mini KPIs derived from df
            mini_kpis = []
            # handle possible column name variations
            cols_lower = [c.lower() for c in df_rg.columns]
            if "sio2" in cols_lower:
                c = df_rg.columns[cols_lower.index("sio2")]
                mini_kpis.append(("SiO₂ (avg)", round(df_rg[c].mean(), 2), None, " %"))
            if "moisture" in df_rg.columns:
                mini_kpis.append(("Moisture (avg)", round(df_rg["moisture"].mean(), 2), None, " %"))
            if "blaine" in cols_lower or "blain" in cols_lower:
                if "blaine" in df_rg.columns:
                    col_b = "blaine"
                else:
                    col_b = [c for c in df_rg.columns if c.lower() == "blain"][0]
                mini_kpis.append(("Blaine (avg)", int(df_rg[col_b].mean()), None, ""))

            if mini_kpis:
                show_kpi_cards(mini_kpis)

            # time-series visuals (use index as time if no time column)
            df_rg_t = df_rg.reset_index().rename(columns={"index":"t"})
            if "raw_material_variability" in df_rg_t.columns:
                draw_plotly_timechart(df_rg_t, "t", "raw_material_variability", title="Raw Material Variability")
            if "grinding_efficiency" in df_rg_t.columns:
                draw_plotly_timechart(df_rg_t, "t", "grinding_efficiency", title="Grinding Efficiency", height=260)

            # scatter relationship
            if {"raw_material_variability", "grinding_efficiency"}.issubset(set(df_rg.columns)):
                fig = px.scatter(df_rg, x="raw_material_variability", y="grinding_efficiency",
                                 title="Variability vs Grinding Efficiency", hover_data=df_rg.columns)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No raw/grinding data — run ingestion (sidebar)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Grinding — Historic")
        if df_rg is not None and "grinding_efficiency" in df_rg.columns:
            mini_sparkline(df_rg, "grinding_efficiency", height=80)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Kiln & Clinker")
        if df_cl is not None:
            if "kiln_temp" in df_cl.columns:
                gauge_metric(df_cl["kiln_temp"].iloc[-1], "Kiln Temp (°C)", min_val=1200, max_val=1600)
            if "oxygen_level" in df_cl.columns:
                st.metric("Oxygen Level", f"{df_cl['oxygen_level'].iloc[-1]} %")
            if "feed_rate" in df_cl.columns:
                st.metric("Feed Rate", f"{df_cl['feed_rate'].iloc[-1]} tph")
            if "kiln_temp" in df_cl.columns:
                mini_sparkline(df_cl, "kiln_temp", height=60)
        else:
            st.info("No clinker data")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Utilities & Fuel")
        if df_cross is not None:
            if "fuel_calorific" in df_cross.columns:
                df_cross_t = df_cross.reset_index().rename(columns={"index":"t"})
                draw_plotly_timechart(df_cross_t, "t", "fuel_calorific", title="Fuel Calorific", height=200)
            if "predicted_energy" in df_cross.columns:
                st.metric("Predicted Energy (last)", f"{df_cross['predicted_energy'].iloc[-1]:.2f} kWh/t")
        else:
            st.write("Utilities metrics not available")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    show_ai_recommendation("Optimize grinding load distribution (Predicted Energy: ~9.6 kWh/t)",
                            details="Suggested scenario: adjust mill feed profile + increase grinding efficiency by 5%.")

    st.markdown("---")
    st.subheader("Predicted Impact from AI Optimization")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("Energy Reduction")
        st.progress(10)
        st.caption("10% estimated")
    with c2:
        st.write("CO₂ Footprint Reduction")
        st.progress(12)
        st.caption("12% estimated")
    with c3:
        st.write("Higher Alt Fuel Usage")
        st.progress(20)
        st.caption("20% possible uplift")
    with c4:
        st.write("Production Stability")
        st.progress(6)
        st.caption("6% improvement")

    st.markdown("---")
    st.subheader("Alerts & Events")
    st.info("High kiln burner fluctuation detected — suggest adjust primary airflow")
    st.warning("RDF calorific variance logged — suggest sampling adjustment")

    st.markdown("---")
    st.subheader("Audit Timeline")
    st.write("- 10:04 — Simulated scenario")
    st.write("- 10:12 — Operator input")
    st.write("- 10:18 — Change applied")

# -------------------------------------------------------------------
# Raw & Grinding page
# -------------------------------------------------------------------
if menu == "Raw & Grinding":
    st.header("Optimize Raw Materials & Grinding (Interactive)")
    df = load_csv("raw_grinding.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        left, right = st.columns([2, 1])
        with left:
            v = st.slider("Raw Material Variability", 0.0, 0.6, 0.2, step=0.01)
            e = st.slider("Grinding Efficiency", 0.5, 1.0, 0.85, step=0.01)
        with right:
            st.markdown("#### Scenario Controls")
            st.write("Adjust variability and efficiency to see local model prediction")

        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            if st.button("Predict (local)"):
                with st.spinner("Training small local model and predicting..."):
                    path = train_raw_grinding(n=300)
                    m = joblib.load(path)
                    pred = m.predict([[v, e]])[0]
                    st.success(f"Predicted energy consumption: {pred:.2f} kWh/t")
        with col_pred2:
            if st.button("Trigger Vertex Batch (placeholder)"):
                st.info("This would submit a Vertex AI batch prediction job (placeholder).")

        # Scatter + rolling mean
        if {"raw_material_variability", "grinding_efficiency"}.issubset(set(df.columns)):
            fig = px.scatter(df, x="raw_material_variability", y="grinding_efficiency",
                             title="Variability vs Grinding Efficiency", hover_data=df.columns)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient columns for historic visual.")

# -------------------------------------------------------------------
# Clinker page
# -------------------------------------------------------------------
if menu == "Clinker":
    st.header("Balance Clinkerization Parameters")
    df = load_csv("clinker.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        temp = st.slider("Kiln Temp (°C)", 1400, 1500, 1450)
        feed = st.slider("Feed Rate (tph)", 250, 350, 300)
        oxy = st.slider("Oxygen Level (%)", 2.0, 5.0, 3.5)

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Predict (local)"):
                with st.spinner("Training local clinker model..."):
                    path = train_clinker(n=300)
                    m = joblib.load(path)
                    pred = m.predict([[temp, feed, oxy]])[0]
                    st.success(f"Predicted energy use: {pred:.2f}")
            if "kiln_temp" in df.columns:
                df_t = df.reset_index().rename(columns={"index":"t"})
                draw_plotly_timechart(df_t, "t", "kiln_temp", "Kiln Temp (°C)", height=360)
        with col2:
            if "oxygen_level" in df.columns:
                st.metric("Oxygen Level (last)", f"{df['oxygen_level'].iloc[-1]} %")
            if "feed_rate" in df.columns:
                st.metric("Feed Rate (last)", f"{df['feed_rate'].iloc[-1]} tph")

# -------------------------------------------------------------------
# Quality page
# -------------------------------------------------------------------
if menu == "Quality":
    st.header("Ensure Quality Consistency")
    df = load_csv("quality.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        si = st.number_input("SiO2 %", 10.0, 30.0, 21.0)
        moist = st.number_input("Moisture %", 0.1, 10.0, 4.0)
        bl = st.number_input("Blaine", 200, 500, 330)
        if st.button("Predict (local)"):
            with st.spinner("Training quality model..."):
                path = train_quality(n=300)
                m = joblib.load(path)
                pred = m.predict([[si, moist, bl]])[0]
                st.success(f"Predicted compressive strength: {pred:.2f} MPa")

        # Distribution of Blaine if available
        col_b = None
        if df is not None:
            cols_lower = [c.lower() for c in df.columns]
            if "blaine" in cols_lower:
                col_b = df.columns[cols_lower.index("blaine")]
            elif "blain" in cols_lower:
                col_b = df.columns[cols_lower.index("blain")]
        if col_b:
            fig = px.histogram(df, x=col_b, nbins=30, title="Blaine Distribution")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Alt Fuel page
# -------------------------------------------------------------------
if menu == "Alt Fuel":
    st.header("Maximize Alternative Fuel Use (TSR)")
    df = load_csv("altfuel.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        fuel = st.number_input("Fuel Calorific (kcal/kg)", 1000, 6000, 3500)
        rfd = st.slider("RDF share", 0.0, 0.8, 0.3, step=0.01)
        tsr = st.slider("Current TSR %", 0.0, 60.0, 32.0)
        if st.button("Predict (local)"):
            with st.spinner("Training altfuel model..."):
                path = train_altfuel(n=300)
                m = joblib.load(path)
                pred = m.predict([[fuel, rfd, tsr]])[0]
                st.success(f"Predicted energy consumption: {pred:.2f}")

        if df is not None and {"fuel_calorific", "tsr"}.issubset(set(df.columns)):
            d = df.reset_index().rename(columns={"index":"t"})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["t"], y=d["fuel_calorific"], fill='tozeroy', name='Fuel Calorific'))
            fig.add_trace(go.Scatter(x=d["t"], y=d["tsr"], fill='tozeroy', name='TSR'))
            fig.update_layout(title="Fuel Calorific & TSR", height=360, margin=dict(l=10,r=10,t=40,b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Cross page
# -------------------------------------------------------------------
if menu == "Cross":
    st.header("Strategic Cross-Process Optimization")
    df = load_csv("cross.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        if st.button("Predict (local)"):
            with st.spinner("Training cross model..."):
                path = train_cross(n=300)
                m = joblib.load(path)
                try:
                    row = df.tail(1)
                    X = row[
                        [
                            "raw_material_variability",
                            "grinding_efficiency",
                            "kiln_temp",
                            "feed_rate",
                            "oxygen_level",
                            "tsr",
                            "fuel_calorific",
                        ]
                    ].values[0].tolist()
                except Exception:
                    X = [0.2, 0.85, 1450, 300, 3.5, 32.0, 3500.0]
                pred = m.predict([X])[0]
                st.success(f"Cross predicted energy: {pred:.2f}")

        last = None
        try:
            last = df.tail(1)
        except Exception:
            last = None
        if last is not None and not last.empty:
            metrics = []
            labels = []
            for col in ["raw_material_variability", "grinding_efficiency", "kiln_temp", "feed_rate", "oxygen_level", "tsr", "fuel_calorific"]:
                if col in last.columns:
                    metrics.append(float(last[col].values[0]))
                    labels.append(col.replace("_", " ").title())
            if metrics:
                radar_chart(metrics, labels, title="Last Snapshot")

# -------------------------------------------------------------------
# Config page
# -------------------------------------------------------------------
if menu == "Config":
    st.header("Edit and persist configuration (BigQuery)")
    st.write("Requires GCP ADC credentials and dataset.table config exists.")
    proc = st.selectbox("Process", ["RM", "Clinker", "Quality", "Fuel", "Cross"])
    pname = st.text_input("Parameter name", "tsr_threshold")
    pval = st.text_input("Parameter value", "0.2")
    if st.button("Save to BigQuery"):
        try:
            write_config_to_bq(proc, pname, pval)
            st.success("Saved config to BigQuery")
        except Exception as ex:
            st.error("Failed: " + str(ex))
    if st.button("Read Configs"):
        try:
            dfc = read_config_from_bq(None)
            st.write("Fetched configs")
            for idx, row in dfc.iterrows():
                st.write(f"- **{row['process']}**: {row['param_name']} = {row['param_value']}")
        except Exception as ex:
            st.error("Failed: " + str(ex))

# -------------------------------------------------------------------
# Train page
# -------------------------------------------------------------------
if menu == "Train":
    st.header("Train and optionally register model")
    if st.button("Train All Local Models"):
        with st.spinner("Training all..."):
            train_raw_grinding()
            train_clinker()
            train_quality()
            train_altfuel()
            train_cross()
            st.success("All local models trained and saved to /models")

    if st.button("Upload & register Raw Grinding model to Vertex (requires GCP)"):
        try:
            path = os.path.join(os.path.dirname(__file__), "..", "models", "raw_grinding.joblib")
            res = upload_and_register(path, "raw-grinding-demo")
            st.write(res)
        except Exception as ex:
            st.error("Register failed: " + str(ex))

# Footer
st.markdown("---")
#st.caption("GenAI Cement demo — Streamlit UI with Plotly visuals. For production, wire in real telemetry and secure GCP credentials.")
