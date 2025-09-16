import streamlit as st
import pandas as pd
import numpy as np
# from data_ingestion import load_sensor_data
from sklearn.linear_model import LinearRegression
from ai_engine import train_model, predict_energy

# st.title("GenAI Cement - Operator Dashboard")
st.set_page_config(page_title="GenAI Cement - Operator Dashboard", layout="wide")

# Sidebar navigation
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

# Page title
st.title("GenAI Cement — Operator Dashboard")
st.caption("Plant: Enclave–1")

# Load data
# df = pd.DataFrame({
#     "Process": ["Grinding", "Clinkerization", "Fuel Use", "Utilities"],
#     "energy_consumption": [120, 300, 180, 90],
#     "Carbon_Emissions": [12, 35, 18, 8],
#     # Add columns for AI model to avoid KeyError
#     "raw_material_variability": [0.2, 0.3, 0.1, 0.25],
#     "grinding_efficiency": [0.85, 0.9, 0.8, 0.88]
# })
# st.line_chart(df[['energy_consumption']])
# -------------------------------------------------------------------
# Demo Dataset
# -------------------------------------------------------------------
np.random.seed(42)
df = pd.DataFrame({
    "raw_material_variability": np.random.normal(0.2, 0.05, 50),
    "grinding_efficiency": np.random.normal(0.8, 0.1, 50),
    "energy_consumption": np.random.normal(9.6, 0.4, 50)
})

# -------------------------------------------------------------------
# KPIs
# -------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Energy Intensity", f"{df['energy_consumption'].mean():.2f} kWh/t", "-8%")
k2.metric("Thermal Substitution Rate", "32%", "+12%")
k3.metric("Off-spec Events", "2 / month", "-45%")
k4.metric("CO₂ Emissions (Scope 1)", "0.82 t/ton", "-10%")

st.markdown("---")

# -------------------------------------------------------------------
# Plant Overview section
# -------------------------------------------------------------------
st.subheader("Plant Overview — Real-time")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### Raw Materials")
    st.write("SiO₂: 21.4% | C60: 92.1%")
    st.write("Moisture: 4.2% | Blaine: 330")
    st.line_chart(df["raw_material_variability"])

with col2:
    st.markdown("#### Grinding (Mill 2)")
    st.write("Power: 4200 kW | Load: 78%")
    st.write("PSD: 12–43 µm | Separator: 65%")
    st.line_chart(df["grinding_efficiency"])

with col3:
    st.markdown("#### Kiln & Clinker")
    st.write("Burner Temp: 1480°C")
    st.write("TSR: 32% | CO₂: 9.92 trion")
    st.write("NOx: OK")

with col4:
    st.markdown("#### Utilities")
    st.write("Compressors: 1250")
    st.write("Pumps & Steam: 400")
    st.write("Material Handling: 78%")
    st.line_chart(df["energy_consumption"])

st.markdown("---")

# -------------------------------------------------------------------
# AI Recommendations Section
# -------------------------------------------------------------------
st.subheader("AI Recommendations")
st.radio("Mode", ["Recommend", "Autonomous"], horizontal=True, index=1)

# Train model with demo data
model = train_model(df)

# Pick last row as "latest sensor input"
latest = df.iloc[-1][["raw_material_variability", "grinding_efficiency"]].values.reshape(1, -1)
pred = predict_action(model, latest)

st.info(f"🔧 Predicted Action: {pred}")
st.write("ETA: 16 min")

st.success("💡 Scenario Simulator: Adjust wet feed spike + optimize RDF fuel mix → Lower emissions")

a1, a2 = st.columns(2)
a1.button("Explanation")
a2.button("Approve & Apply")

st.markdown("---")

# -------------------------------------------------------------------
# Alerts
# -------------------------------------------------------------------
st.subheader("Alerts & Events")
st.error("⚠️ High Kiln burner fluctuation detected — Suggest: Adjust primary airflow")
st.warning("⚠️ RDF calorific variance logged — Suggest: Sampling adjustment")

st.markdown("---")

# -------------------------------------------------------------------
# Audit & Controls
# -------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Audit Timeline")
    st.write("10:04 — Simulated scenario")
    st.write("10:12 — Operator input")
    st.write("10:18 — Change applied")

with c2:
    st.markdown("#### Quick Controls")
    st.write("Auto-mode threshold: Confidence ≥ 85%")
    st.write("Safety limits: Active ✅")

# Train model on demo dataset
def train_model(data):
    X = data[['raw_material_variability', 'grinding_efficiency']]
    y = data['energy_consumption']
    model = LinearRegression()
    model.fit(X, y)
    return model
# Simple wrapper for prediction → returns readable recommendation
def predict_action(model, latest_input):
    try:
        pred = model.predict(latest_input)[0]
    except Exception:
        return "⚠️ Prediction failed — using fallback suggestion."

    # Demo recommendation logic
    if pred > 10:
        return f"Reduce mill separator speed by 3% (Predicted Energy: {pred:.2f} kWh/t)"
    elif pred > 9:
        return f"Optimize grinding load distribution (Predicted Energy: {pred:.2f} kWh/t)"
    else:
        return f"Maintain current settings (Predicted Energy: {pred:.2f} kWh/t)"

# # Train model
# model = train_model(df)

# # User input
# st.subheader("Predict Energy Consumption")
# variability = st.slider("Raw Material Variability", 0.05, 0.2, 0.12)
# efficiency = st.slider("Grinding Efficiency", 0.75, 0.95, 0.85)

# pred = predict_energy(model, variability, efficiency)
# st.write(f"🔮 Predicted Energy Consumption: {pred:.2f} units")
