# ui/streamlit_app.py
import sys
import os
from pathlib import Path
import time
import joblib

# Ensure repo root is on sys.path so sibling modules in src/ can be imported.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# NOTE: these imports expect the modules to be at repo_root/src/*.py
# and that those files use top-level imports (no leading dot).
from data_generator import gen_raw_grinding, gen_clinker, gen_quality, gen_altfuel, gen_cross
from trainers import (
    train_raw_grinding,
    train_clinker,
    train_quality,
    train_altfuel,
    train_cross,
    upload_and_register,
)
from ingest_simulator import write_all as ingest_write_all
from gcp_utils import write_config_to_bq, read_config_from_bq

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    layout="wide",
    page_title="GenAI Cement - Operator Dashboard",
    initial_sidebar_state="expanded",
)

# ---------- Paths ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_csv(name):
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

# ---------- Small helper visual components ----------
def metric_card(label, value, delta=None, help_text=None, width=3):
    st.metric(label=label, value=value, delta=delta)
    if help_text:
        st.caption(help_text)

def draw_line_chart(df, x_col, y_col, title=None, height=150):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        st.write("No data to show")
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(x=alt.X(x_col, axis=alt.Axis(labelAngle=0)), y=alt.Y(y_col))
        .properties(height=height, width=None, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def show_top_action(pred_text, details=None):
    with st.container():
        st.markdown("### 🔮 AI Recommendations")
        st.success(pred_text)
        if details:
            st.write(details)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Overview", "Raw & Grinding", "Clinker", "Quality", "Alt Fuel", "Cross", "Config", "Train"],
)

st.sidebar.markdown("---")
st.sidebar.write("Developer controls")
if st.sidebar.checkbox("Regenerate synthetic CSVs (ingest)"):
    ingest_write_all()
    st.sidebar.success("Regenerated synthetic CSVs in `data/`")

# ---------- Top-level Dashboard (Overview) ----------
if menu == "Overview":
    st.title("GenAI Cement — Operator Dashboard")
    # Top KPI metrics (placeholder values read from data if present)
    # Try to compute a few KPIs from data if available
    df_rg = load_csv("raw_grinding.csv")
    df_cl = load_csv("clinker.csv")
    df_q = load_csv("quality.csv")
    df_af = load_csv("altfuel.csv")
    df_cross = load_csv("cross.csv")

    # Fallback sample values
    energy_intensity = (
        round(df_cross["predicted_energy"].mean(), 2) if (df_cross is not None and "predicted_energy" in df_cross.columns) else 9.58
    )
    tsr = round(df_af["tsr"].mean(), 1) if (df_af is not None and "tsr" in df_af.columns) else 32
    off_spec = 2  # placeholder
    co2 = 0.82

    # metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Energy Intensity", f"{energy_intensity} kWh/t", delta="-0.4")
    with col2:
        st.metric("Thermal Substitution Rate", f"{tsr} %")
    with col3:
        st.metric("Off-spec Events", f"{off_spec} / month")
    with col4:
        st.metric("CO₂ Emissions", f"{co2} t/ton")

    st.markdown("---")

    # Plant overview charts
    st.subheader("Plant Overview — Real-time")
    r1, r2 = st.columns([2, 1])
    with r1:
        st.markdown("#### Raw Materials")
        if df_rg is not None:
            st.dataframe(df_rg.head(6))
            draw_line_chart(df_rg.reset_index().rename(columns={"index": "t"}), "t", "raw_material_variability", "Raw Material Variability")
        else:
            st.info("No raw/grinding data — run ingestion (Overview sidebar)")

        st.markdown("#### Grinding (Mill 2)")
        if df_rg is not None and "grinding_efficiency" in df_rg.columns:
            draw_line_chart(df_rg.reset_index().rename(columns={"index": "t"}), "t", "grinding_efficiency", "Grinding Efficiency")
    with r2:
        st.markdown("#### Kiln & Clinker")
        if df_cl is not None:
            st.dataframe(df_cl.head(4))
            draw_line_chart(df_cl.reset_index().rename(columns={"index": "t"}), "t", "kiln_temp", "Kiln Temp (°C)")
        else:
            st.info("No clinker data")

        st.markdown("#### Utilities")
        # Combined simple plot from cross or fallback
        if df_cross is not None and "fuel_calorific" in df_cross.columns:
            draw_line_chart(df_cross.reset_index().rename(columns={"index": "t"}), "t", "fuel_calorific", "Fuel Calorific")
        else:
            st.write("Utilities metrics not available")

    st.markdown("---")

    # AI Recommendation & impact
    top_action = "Optimize grinding load distribution (Predicted Energy: ~9.6 kWh/t)"
    show_top_action(top_action, details="Suggested scenario: adjust mill feed profile + increase grinding efficiency by 5%.")

    # Predicted impacts (fake progress bars)
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
    # Display sample alerts (if you have an alerts CSV you can load)
    st.info("High kiln burner fluctuation detected — suggest adjust primary airflow")
    st.warning("RDF calorific variance logged — suggest sampling adjustment")
    st.markdown("---")

    st.subheader("Audit Timeline")
    st.write("- 10:04 — Simulated scenario")
    st.write("- 10:12 — Operator input")
    st.write("- 10:18 — Change applied")

# ---------- Raw & Grinding page ----------
if menu == "Raw & Grinding":
    st.header("Optimize Raw Materials & Grinding")
    df = load_csv("raw_grinding.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        st.dataframe(df.head(8))

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

    # Visualize historic relationship if df exists
    if df is not None and {"raw_material_variability", "grinding_efficiency"}.issubset(set(df.columns)):
        st.markdown("### Historic relationship")
        # simple scatter chart
        chart = (
            alt.Chart(df)
            .mark_circle(size=40, opacity=0.5)
            .encode(x="raw_material_variability", y="grinding_efficiency", tooltip=["raw_material_variability", "grinding_efficiency"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- Clinker page ----------
if menu == "Clinker":
    st.header("Balance Clinkerization Parameters")
    df = load_csv("clinker.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        st.dataframe(df.head(8))

    temp = st.slider("Kiln Temp (°C)", 1400, 1500, 1450)
    feed = st.slider("Feed Rate (tph)", 250, 350, 300)
    oxy = st.slider("Oxygen Level (%)", 2.0, 5.0, 3.5)

    if st.button("Predict (local)"):
        with st.spinner("Training local clinker model..."):
            path = train_clinker(n=300)
            m = joblib.load(path)
            pred = m.predict([[temp, feed, oxy]])[0]
            st.success(f"Predicted energy use: {pred:.2f}")

# ---------- Quality page ----------
if menu == "Quality":
    st.header("Ensure Quality Consistency")
    df = load_csv("quality.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        st.dataframe(df.head(8))

    si = st.number_input("SiO2 %", 10.0, 30.0, 21.0)
    moist = st.number_input("Moisture %", 0.1, 10.0, 4.0)
    bl = st.number_input("Blaine", 200, 500, 330)

    if st.button("Predict (local)"):
        with st.spinner("Training quality model..."):
            path = train_quality(n=300)
            m = joblib.load(path)
            pred = m.predict([[si, moist, bl]])[0]
            st.success(f"Predicted compressive strength: {pred:.2f} MPa")

# ---------- Alt Fuel page ----------
if menu == "Alt Fuel":
    st.header("Maximize Alternative Fuel Use (TSR)")
    df = load_csv("altfuel.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        st.dataframe(df.head(8))

    fuel = st.number_input("Fuel Calorific (kcal/kg)", 1000, 6000, 3500)
    rfd = st.slider("RDF share", 0.0, 0.8, 0.3, step=0.01)
    tsr = st.slider("Current TSR %", 0.0, 60.0, 32.0)

    if st.button("Predict (local)"):
        with st.spinner("Training altfuel model..."):
            path = train_altfuel(n=300)
            m = joblib.load(path)
            pred = m.predict([[fuel, rfd, tsr]])[0]
            st.success(f"Predicted energy consumption: {pred:.2f}")

# ---------- Cross page ----------
if menu == "Cross":
    st.header("Strategic Cross-Process Optimization")
    df = load_csv("cross.csv")
    if df is None:
        st.warning("No data found; run ingestion from Overview.")
    else:
        st.dataframe(df.head(6))

    if st.button("Predict (local)"):
        with st.spinner("Training cross model..."):
            path = train_cross(n=300)
            m = joblib.load(path)
            # fallback if columns missing
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
                # fallback default vector
                X = [0.2, 0.85, 1450, 300, 3.5, 32.0, 3500.0]
            pred = m.predict([X])[0]
            st.success(f"Cross predicted energy: {pred:.2f}")

# ---------- Config page ----------
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
            st.dataframe(dfc)
        except Exception as ex:
            st.error("Failed: " + str(ex))

# ---------- Train page ----------
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

# ---------- Footer / debug info ----------
st.markdown("---")
st.caption("GenAI Cement demo — Streamlit UI. For production, wire in real telemetry and secure GCP credentials.")
