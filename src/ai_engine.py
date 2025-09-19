# ai_engine.py
import numpy as np
from sklearn.linear_model import LinearRegression

# --------------------------
# Train model
# --------------------------
def train_model(data):
    X = data[['raw_material_variability', 'grinding_efficiency']]
    y = data['energy_consumption']
    model = LinearRegression()
    model.fit(X, y)
    return model

# --------------------------
# Predict action (recommendation text)
# --------------------------
def predict_action(model, latest_input):
    try:
        pred = model.predict(latest_input)[0]
    except Exception:
        return "⚠️ Prediction failed — using fallback suggestion."

    if pred > 10:
        return f"Reduce mill separator speed by 3% (Predicted Energy: {pred:.2f} kWh/t)"
    elif pred > 9:
        return f"Optimize grinding load distribution (Predicted Energy: {pred:.2f} kWh/t)"
    else:
        return f"Maintain current settings (Predicted Energy: {pred:.2f} kWh/t)"

# --------------------------
# Predict energy (numeric value for slider demo)
# --------------------------
def predict_energy(model, variability, efficiency):
    arr = np.array([[variability, efficiency]])
    return model.predict(arr)[0]
