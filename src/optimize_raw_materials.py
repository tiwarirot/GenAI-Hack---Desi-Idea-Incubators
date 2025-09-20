import os
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import bigquery, storage, aiplatform
import tensorflow as tf
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from threading import Thread
import streamlit as st

# ---------------------------
# GCP Config
# ---------------------------
PROJECT_ID = "my-project-48035-282812"
BUCKET = "optimize_raw_materials"
BQ_DATASET = "cement_ops"
BQ_TABLE = "raw_materials"
REGION = "us-central1"
FEATURE_STORE_ID = "cement_features"
MODEL_DISPLAY_NAME = "cement-tft-model"
API_IMAGE = f"gcr.io/{PROJECT_ID}/cement-optimizer-api"

# Toggle mode
USE_GCP = False   # True = pipeline simulation, False = local prototype

# ---------------------------
# Step 1. Streaming ingestion via Dataflow
# ---------------------------
def run_dataflow_job(input_topic, output_table):
    """Ingest streaming data (simulated SCADA) into BigQuery."""
    options = PipelineOptions(
        runner="DataflowRunner",
        project=PROJECT_ID,
        region=REGION,
        temp_location=f"gs://{BUCKET}/tmp",
        streaming=True,
    )
    with beam.Pipeline(options=options) as p:
        (
            p
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(topic=input_topic)
            | "ParseJSON" >> beam.Map(json.loads)
            | "WriteToBQ" >> beam.io.WriteToBigQuery(
                output_table,
                schema="raw_material_variability:FLOAT, grinding_efficiency:FLOAT, energy_consumption:FLOAT, timestamp:TIMESTAMP",
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            )
        )

# ---------------------------
# Step 2. Store Raw Data (Cloud Storage + BigQuery)
# ---------------------------
def export_to_gcs():
    client = bigquery.Client()
    query = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    df = client.query(query).to_dataframe()

    local_file = "cement_data.csv"
    df.to_csv(local_file, index=False)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob("data/cement_data.csv")
    blob.upload_from_filename(local_file)
    print("Exported data to GCS")

# ---------------------------
# Step 3. Register Features in Vertex AI Feature Store
# ---------------------------
def create_featurestore():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    featurestore = aiplatform.Featurestore.create(
        featurestore_id=FEATURE_STORE_ID,
        online_store_fixed_node_count=1,
    )
    entity_type = featurestore.create_entity_type(entity_type_id="cement_batch")
    entity_type.create_feature("raw_material_variability", value_type="FLOAT64")
    entity_type.create_feature("grinding_efficiency", value_type="FLOAT64")
    entity_type.create_feature("energy_consumption", value_type="FLOAT64")
    print("Featurestore created")

# ---------------------------
# Step 4. Train Model (local demo version)
# ---------------------------
MODEL_PATH = "cement_model.h5"

def train_model_local():
    X = np.array([
        [0.2, 0.8],
        [0.3, 0.75],
        [0.25, 0.9],
        [0.15, 0.85],
        [0.35, 0.7]
    ])
    y = np.array([100, 120, 95, 90, 130])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=100, verbose=0)
    model.save(MODEL_PATH)

    return model

if not os.path.exists(MODEL_PATH):
    train_model_local()

local_model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------
# Optimization Logic
# ---------------------------
def optimize_inputs(rm_var, grind_eff, model):
    X = np.array([[rm_var, grind_eff]])
    baseline_energy = model.predict(X, verbose=0).tolist()[0][0]

    optimized_rm_var = max(0.1, rm_var * 0.9)
    optimized_grind_eff = min(0.95, grind_eff * 1.05)

    X_opt = np.array([[optimized_rm_var, optimized_grind_eff]])
    optimized_energy = model.predict(X_opt, verbose=0).tolist()[0][0]

    energy_reduction = (baseline_energy - optimized_energy) / baseline_energy * 100

    return {
        "baseline": {"rm_var": rm_var, "grind_eff": grind_eff, "energy": baseline_energy},
        "optimized": {"rm_var": optimized_rm_var, "grind_eff": optimized_grind_eff, "energy": optimized_energy},
        "impacts": {
            "energy_reduction_%": round(energy_reduction, 2),
            "co2_reduction_%": round(energy_reduction * 0.9, 2),
            "alt_fuel_usage_%": 20.0,
            "stability_%": 6.0
        }
    }

# ---------------------------
# Step 5. Flask API (Cloud Run style)
# ---------------------------
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    rm_var = float(data["raw_material_variability"])
    grind_eff = float(data["grinding_efficiency"])
    result = optimize_inputs(rm_var, grind_eff, local_model)
    return jsonify(result)

def run_flask():
    app.run(host="0.0.0.0", port=8080)

# ---------------------------
# Step 6. Streamlit Dashboard (UI)
# ---------------------------
def run_streamlit():
    st.set_page_config(page_title="GenAI Cement Optimizer", layout="wide")
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", ["Dashboard", "About"])

    if section == "Dashboard":
        st.title("⚙️ Cement Plant Optimization Dashboard")
        rm_var = st.slider("Raw Material Variability", 0.1, 0.5, 0.25, 0.01)
        grind_eff = st.slider("Grinding Efficiency", 0.6, 0.95, 0.8, 0.01)

        if st.button("Optimize"):
            result = optimize_inputs(rm_var, grind_eff, local_model)
            st.metric("Baseline Energy", f"{result['baseline']['energy']:.2f}")
            st.metric("Optimized Energy", f"{result['optimized']['energy']:.2f}")
            st.write("Predicted Impacts", result["impacts"])
    else:
        st.title("About")
        if USE_GCP:
            st.write("Connected to **Google Cloud Pipeline**: Dataflow → BigQuery → Vertex AI → Cloud Run → Looker.")
        else:
            st.write("Running in **local prototype mode** with TensorFlow + Streamlit + Flask.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    thread = Thread(target=run_flask, daemon=True)
    thread.start()
    run_streamlit()
