from src.data_ingestion import load_sensor_data
from src.ai_engine import train_model, predict_energy
from src.integration import send_to_erp

def main():
    df = load_sensor_data()
    model = train_model(df)
    prediction = predict_energy(model, 0.14, 0.84)
    print("Predicted Energy Consumption:", prediction)
    send_to_erp({'energy': prediction, 'status': 'optimized'})

if __name__ == "__main__":
    main()
