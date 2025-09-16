import streamlit as st
import pandas as pd
from data_ingestion import load_sensor_data
from ai_engine import train_model, predict_energy

st.title("GenAI Cement - Operator Dashboard")

# Load data
df = load_sensor_data()
st.line_chart(df[['energy_consumption']])

# Train model
model = train_model(df)

# User input
st.subheader("Predict Energy Consumption")
variability = st.slider("Raw Material Variability", 0.05, 0.2, 0.12)
efficiency = st.slider("Grinding Efficiency", 0.75, 0.95, 0.85)

pred = predict_energy(model, variability, efficiency)
st.write(f"🔮 Predicted Energy Consumption: {pred:.2f} units")
