import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data[['raw_material_variability', 'grinding_efficiency']]
    y = data['energy_consumption']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_energy(model, variability, efficiency):
    return model.predict([[variability, efficiency]])[0]

if __name__ == "__main__":
    from data_ingestion import load_sensor_data
    df = load_sensor_data()
    model = train_model(df)
    print("Predicted energy:", predict_energy(model, 0.14, 0.84))
