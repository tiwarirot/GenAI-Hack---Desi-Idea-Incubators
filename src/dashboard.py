import streamlit as st
import pandas as pd
import numpy as np
from ai_engine import train_model, predict_action

st.set_page_config(page_title="GenAI Cement - Operator Dashboard", layout="wide")

# --------------------------
# Custom CSS for styling
# --------------------------
st.markdown("""
    <style>
    /* General */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
    }
    .section-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 14px;
        color: #666;
    }
    .toggle {
        background-color: #EAF6FF;
        padding: 6px 14px;
        border-radius: 16px;
        font-size: 14px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar navigation
# --------------------------
st.sidebar.title("GenAI Cement")
st.sidebar.markdown("### Navigation")
st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Processes",
        "Raw Materials",
        "Grinding",
        "Clinkerization",
        "Utilities",
        "AI Tools",
        "Settings",
        "Policies & Safety"
    ],
    index=0
)

# --------------------------
# Title
# --------------------------
st.markdown("<h1>GenAI Cement — Operator Dashboard</h1>", unsafe_allow_html=True)
st.caption("Plant: Enclave–1")

# --------------------------
# Demo Dataset
# --------------------------
np.random.seed(42)
df = pd.DataFrame({
    "raw_material_variability": np.random.normal(0.2, 0.05, 50),
    "grinding_efficiency": np.random.normal(0.8, 0.1, 50),
    "energy_consumption": np.random.normal(9.6, 0.4, 50)
})

# --------------------------
# KPIs
# --------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='metric-card'><div class='kpi-value'>{df['energy_consumption'].mean():.2f} kWh/t</div><div class='kpi-label'>Energy Intensity</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='metric-card'><div class='kpi-value'>32%</div><div class='kpi-label'>Thermal Substitution Rate</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='metric-card'><div class='kpi-value'>2 / month</div><div class='kpi-label'>Off-spec Events</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='metric-card'><div class='kpi-value'>0.82 t/ton</div><div class='kpi-label'>CO₂ Emissions (Scope 1)</div></div>", unsafe_allow_html=True)

st.markdown("")

# --------------------------
# Plant Overview
# --------------------------
st.subheader("Plant Overview — Real-time")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='section-card'><h4>Raw Materials</h4><p>SiO₂: 21.4% | C60: 92.1%<br>Moisture: 4.2% | Blaine: 330</p></div>", unsafe_allow_html=True)
    st.line_chart(df["raw_material_variability"])

with col2:
    st.markdown("<div class='section-card'><h4>Grinding (Mill 2)</h4><p>Power: 4200 kW | Load: 78%<br>PSD: 12–43 µm | Separator: 65%</p></div>", unsafe_allow_html=True)
    st.line_chart(df["grinding_efficiency"])

with col3:
    st.markdown("<div class='section-card'><h4>Kiln & Clinker</h4><p>Burner Temp: 1480°C<br>TSR: 32% | CO₂: 9.92 trion<br>NOx: OK</p></div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='section-card'><h4>Utilities</h4><p>Compressors: 1250<br>Pumps & Steam: 400<br>Material Handling: 78%</p></div>", unsafe_allow_html=True)
    st.line_chart(df["energy_consumption"])

# --------------------------
# AI Recommendations
# --------------------------
st.subheader("AI Recommendations")
st.markdown("<span class='toggle'>Mode: Autonomous</span>", unsafe_allow_html=True)

model = train_model(df)
latest = df.iloc[-1][["raw_material_variability", "grinding_efficiency"]].values.reshape(1, -1)
pred = predict_action(model, latest)

st.markdown(f"<div class='section-card'><b>Top Action (Predicted)</b><br>{pred}<br><small>ETA: 16 min</small></div>", unsafe_allow_html=True)

st.success("💡 Scenario Simulator: Adjust wet feed spike + optimize RDF fuel mix → Lower emissions")

# --------------------------
# Alerts & Controls
# --------------------------
st.subheader("Alerts & Events")
st.error("⚠️ High Kiln burner fluctuation detected — Suggest: Adjust primary airflow")
st.warning("⚠️ RDF calorific variance logged — Suggest: Sampling adjustment")

c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='section-card'><h4>Audit Timeline</h4><p>10:04 — Simulated scenario<br>10:12 — Operator input<br>10:18 — Change applied</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='section-card'><h4>Quick Controls</h4><p>Auto-mode threshold: Confidence ≥ 85%<br>Safety limits: Active ✅</p></div>", unsafe_allow_html=True)
