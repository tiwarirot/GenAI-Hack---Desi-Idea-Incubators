import os
import json
import time
import random
from google.cloud import pubsub_v1, storage, aiplatform
import pandas as pd
import numpy as np
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "my-project-48035-282812"
BUCKET = "clinker_params"
TOPIC_ID = "clinker-sensor-stream"
REGION = "asia-south1"
MODEL_DISPLAY_NAME = "clinkerization-tft-model"

# -----------------------------
# Step 1. Simulate Edge Sensor Data → Pub/Sub
# -----------------------------
def publish_sensor_data():
    """Simulate clinkerization edge sensor readings and push to Pub/Sub"""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

    for _ in range(10):  # simulate 10 messages
        data = {
            "kiln_temp": random.uniform(1350, 1450),
            "feed_rate": random.uniform(250, 350),
            "oxygen_level": random.uniform(2, 5),
            "energy_use": random.uniform(120, 180),
            "timestamp": time.time(),
        }
        publisher.publish(topic_path, json.dumps(data).encode("utf-8"))
        print(f"Published: {data}")
        time.sleep(1)


# -----------------------------
# Step 2. Store Data in GCS
# -----------------------------
def save_to_gcs(local_file="clinker_data.csv"):
    """Save simulated sensor dataset into Cloud Storage"""
    df = pd.DataFrame([
        {"kiln_temp": random.uniform(1350, 1450),
         "feed_rate": random.uniform(250, 350),
         "oxygen_level": random.uniform(2, 5),
         "energy_use": random.uniform(120, 180)} for _ in range(100)
    ])
    df.to_csv(local_file, index=False)

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob("data/clinker_data.csv")
    blob.upload_from_filename(local_file)
    print(f"Data uploaded to gs://{BUCKET}/data/clinker_data.csv")


# -----------------------------
# Step 3. Train Model (TFT/LSTM style in TensorFlow)
# -----------------------------
def train_model():
    df = pd.read_csv("clinker_data.csv")

    X = df[["kiln_temp", "feed_rate", "oxygen_level"]].values
    y = df["energy_use"].values

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=16)

    model.save("clinker_model.h5")
    print("✅ Clinkerization model trained & saved.")


# -----------------------------
# Step 4. Explain Results via Gemini
# -----------------------------
def generate_rationale(predicted_energy, kiln_temp, feed_rate, oxygen_level):
    """Use Gemini to generate rationale for operator"""
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Gemini API (Vertex AI Generative Model)
    gemini = aiplatform.gapic.PredictionServiceClient()
    # For simplicity, we’ll just mock rationale here.
    rationale = (
        f"Predicted energy use is {predicted_energy:.2f} kWh.\n"
        f"- Kiln Temp: {kiln_temp}°C\n"
        f"- Feed Rate: {feed_rate} tph\n"
        f"- Oxygen Level: {oxygen_level}%\n\n"
        "Optimization Insight: Increasing oxygen slightly could reduce fuel use "
        "while maintaining clinker quality."
    )
    return rationale


# -----------------------------
# Step 5. Serve Prediction Locally (Cloud Run API)
# -----------------------------
def run_prediction():
    model = tf.keras.models.load_model("clinker_model.h5")

    sample = np.array([[1400, 300, 3.5]])
    pred = model.predict(sample).tolist()[0][0]

    rationale = generate_rationale(pred, 1400, 300, 3.5)
    print("Prediction:", pred)
    print("Rationale:\n", rationale)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Step 1: publish simulated sensor readings
    publish_sensor_data()

    # Step 2: export to GCS
    save_to_gcs()

    # Step 3: train clinkerization optimization model
    train_model()

    # Step 4: run a prediction & rationale
    run_prediction()

