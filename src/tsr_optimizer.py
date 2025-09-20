import os
import time
import json
import random
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# ML libs for a simple demo model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# GCP libs (optional)
try:
    from google.cloud import pubsub_v1
    from google.cloud import bigquery
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except Exception:
    GCP_AVAILABLE = False

# Flask for optional API
from flask import Flask, request, jsonify

# --------------------------
# Config
# --------------------------
USE_GCP = False  # Set True to run with real GCP (must have credentials & libraries)
PROJECT_ID = "my-project-48035-282812"
PUBSUB_TOPIC = "biomass-feed-topic"  # Pub/Sub topic name
BQ_DATASET = "cement_ops"
BQ_TABLE_RAW = "biomass_feed_raw"
BQ_TABLE_OPT = "tsr_optimizations"
GCS_BUCKET = "your-data-bucket"
REGION = "us-central1"
VERTEX_MODEL_DISPLAY = "tsr-optimizer-model"
GEMINI_MODEL = "projects/your-gcp-project/locations/us-central1/publishers/google/models/text-bison@001"  # example

# --------------------------
# 1) Simulate / Publish biomass feed messages
# --------------------------
def simulate_biomass_messages(n=50, publish=False):
    """
    Create simulated biomass feedstock records.
    If publish=True and GCP available, publish messages to Pub/Sub.
    Otherwise return a pandas DataFrame of messages.
    """
    records = []
    for i in range(n):
        rec = {
            "timestamp": int(time.time()),
            "biomass_type": random.choice(["RDF", "SRF", "Agricultural", "WoodPellet"]),
            "moisture_pct": round(random.uniform(5.0, 25.0), 2),
            "calorific_value_kcal_kg": round(random.uniform(2000, 4500), 1),
            "ash_content_pct": round(random.uniform(2.0, 25.0), 2),
            "feed_rate_tph": round(random.uniform(2.0, 20.0), 2),
            "current_tsr_pct": round(random.uniform(5.0, 35.0), 2),
        }
        records.append(rec)

    df = pd.DataFrame(records)

    if publish and USE_GCP and GCP_AVAILABLE:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        for _, row in df.iterrows():
            publisher.publish(topic_path, json.dumps(row.to_dict()).encode("utf-8"))
        print(f"[Pub/Sub] Published {len(df)} messages to {PUBSUB_TOPIC}")
    else:
        print(f"[Simulate] Generated {len(df)} records (not published).")
    return df

# --------------------------
# 2) Ingest to BigQuery (or local storage)
# --------------------------
def ingest_to_bigquery(df: pd.DataFrame):
    """
    Insert DataFrame rows into BigQuery table BQ_DATASET.BQ_TABLE_RAW.
    If GCP unavailable, save CSV locally.
    """
    if USE_GCP and GCP_AVAILABLE:
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RAW}"
        rows = df.to_dict(orient="records")
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            print("[BigQuery] Insert errors:", errors)
        else:
            print(f"[BigQuery] Inserted {len(rows)} rows into {table_id}")
        return True
    else:
        local_path = "biomass_feed_raw_local.csv"
        df.to_csv(local_path, index=False)
        print(f"[Local] Saved raw feed to {local_path}")
        return False

def read_from_bigquery(limit=1000) -> pd.DataFrame:
    """
    Read recent raw feed data from BigQuery. If not available, read local CSV if present.
    """
    if USE_GCP and GCP_AVAILABLE:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_RAW}` ORDER BY timestamp DESC LIMIT {limit}"
        df = client.query(query).to_dataframe()
        return df
    else:
        local_path = "biomass_feed_raw_local.csv"
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        else:
            # fallback: simulate small dataset
            return simulate_biomass_messages(n=100, publish=False)

# --------------------------
# 3) Feature engineering & local model training (demo)
# --------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw biomass feed data into model-ready features.
    Simple example features:
      - moisture_pct
      - calorific_value_kcal_kg
      - ash_content_pct
      - feed_rate_tph
      - current_tsr_pct
    """
    feat = df[["moisture_pct", "calorific_value_kcal_kg", "ash_content_pct", "feed_rate_tph", "current_tsr_pct"]].copy()
    # Example derived features
    feat["cv_per_moisture"] = feat["calorific_value_kcal_kg"] / (1 + feat["moisture_pct"]/100.0)
    # fill nans
    feat = feat.fillna(feat.mean())
    return feat

def train_local_model(df: pd.DataFrame):
    """
    Train a demo model that predicts energy benefit (negative energy) or directly predicts
    achievable TSR uplift. For demo, we'll predict 'achievable_tsr_pct' as target synthesized.
    """
    # For demo create synthetic target: achievable_tsr = current_tsr + function(calorific, moisture, ash)
    df = df.copy()
    df["achievable_tsr_pct"] = df["current_tsr_pct"] + (df["calorific_value_kcal_kg"] - 2500)/2000*5 - (df["moisture_pct"]/25)*3 - (df["ash_content_pct"]/25)*2
    df["achievable_tsr_pct"] = df["achievable_tsr_pct"].clip(0, 100)

    X = prepare_features(df)
    y = df["achievable_tsr_pct"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("[LocalModel] Trained RandomForestRegressor demo model")
    return model

# --------------------------
# 4) (Optional) Vertex AI training placeholder
# --------------------------
def train_on_vertex_ai(gcs_csv_path: str):
    """
    Placeholder to show how you'd call Vertex AI to train a TFT/LSTM model using your CSV in GCS.
    To actually run, you must implement training code as a custom training job or use AutoML
    and provide dataset to Vertex AI.
    """
    if not (USE_GCP and GCP_AVAILABLE):
        print("[VertexAI] Skipping Vertex AI training (GCP disabled).")
        return None

    aiplatform.init(project=PROJECT_ID, location=REGION)
    # Pseudocode / placeholder: create dataset, submit training job...
    print("[VertexAI] (placeholder) Would submit training job here using", gcs_csv_path)
    # Return a reference to the trained model endpoint
    return None

# --------------------------
# 5) Gemini rationale generation (placeholder)
# --------------------------
def generate_gemini_rationale(inputs: Dict[str, Any], predicted_tsr: float) -> str:
    """
    Use Vertex AI Generative (Gemini) to produce a human-readable rationale.
    In demo mode this returns a mocked explanation. If USE_GCP and configured, you can call
    the Vertex AI Text generation APIs here.
    """
    if USE_GCP and GCP_AVAILABLE:
        try:
            aiplatform.init(project=PROJECT_ID, location=REGION)
            # Example of calling text-bison via aiplatform: (requires correct SDK use)
            # For brevity we provide a mocked call here. Replace with real code as needed.
            prompt = (
                "You are an operations expert. Given biomass feedstock metrics and a predicted TSR, "
                f"generate an operator-ready rationale. Inputs: {json.dumps(inputs)}, predicted_tsr: {predicted_tsr:.2f}%"
            )
            # Real call would be something like aiplatform.TextGenerationModel.from_pretrained(...).predict(...)
            # For now, return a simple formatted explanation:
        except Exception as e:
            print("[Gemini] client error:", e)
    # Mocked rationale:
    rationale = (
        f"Predicted achievable TSR: {predicted_tsr:.2f}%.\n"
        f"Rationale: Feedstock CV={inputs['calorific_value_kcal_kg']} kcal/kg and moisture={inputs['moisture_pct']}% "
        f"indicate good fuel value; ash {inputs['ash_content_pct']}% is moderate. "
        "We recommend increasing TSR gradually and monitor flame stability and clinker quality."
    )
    return rationale

# --------------------------
# 6) Optimization logic for a single feed record
# --------------------------
def optimize_tsr_for_record(model, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a trained model and a single biomass feed record, predict achievable TSR and recommended setpoint.
    """
    df = pd.DataFrame([record])
    X = prepare_features(df)
    predicted_tsr = model.predict(X)[0]  # predicted achievable TSR %
    # Proposed setpoint: push TSR towards predicted achievable but limited by safety rules
    current = record.get("current_tsr_pct", 10.0)
    recommended_tsr = min(predicted_tsr, current + 10.0)  # don't increase more than +10% in one step
    # Estimate impacts (simple heuristic)
    energy_reduction_pct = (recommended_tsr - current) * 0.4  # example: each 1% TSR gives 0.4% energy reduction
    co2_reduction_pct = energy_reduction_pct * 0.9
    alt_fuel_usage_pct = recommended_tsr  # placeholder: TSR itself = alt fuel %
    stability_improvement_pct = min(10.0, (recommended_tsr - current) * 0.5)

    rationale = generate_gemini_rationale(record, predicted_tsr)

    return {
        "timestamp": int(time.time()),
        "biomass_type": record.get("biomass_type"),
        "current_tsr_pct": current,
        "predicted_achievable_tsr_pct": float(round(predicted_tsr, 2)),
        "recommended_tsr_pct": float(round(recommended_tsr, 2)),
        "energy_reduction_pct": float(round(energy_reduction_pct, 2)),
        "co2_reduction_pct": float(round(co2_reduction_pct, 2)),
        "alt_fuel_usage_pct": float(round(alt_fuel_usage_pct, 2)),
        "stability_improvement_pct": float(round(stability_improvement_pct, 2)),
        "rationale": rationale
    }

# --------------------------
# 7) Persist optimizations to BigQuery (or CSV for local demo)
# --------------------------
def persist_optimizations(records: List[Dict[str, Any]]):
    """
    Write optimization outputs to BigQuery table BQ_TABLE_OPT.
    If GCP not available, write to local CSV 'tsr_optimizations_local.csv'.
    """
    df = pd.DataFrame(records)
    if USE_GCP and GCP_AVAILABLE:
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE_OPT}"
        errors = client.insert_rows_json(table_id, df.to_dict(orient="records"))
        if errors:
            print("[BigQuery] Errors writing optimizations:", errors)
        else:
            print(f"[BigQuery] Persisted {len(df)} optimization rows to {table_id}")
    else:
        local_file = "tsr_optimizations_local.csv"
        if os.path.exists(local_file):
            existing = pd.read_csv(local_file)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(local_file, index=False)
        print(f"[Local] Saved optimizations to {local_file}")

# --------------------------
# 8) End-to-end runner
# --------------------------
def run_pipeline_demo(n_messages=50):
    # 1) simulate messages
    raw_df = simulate_biomass_messages(n=n_messages, publish=False)
    # 2) ingest to bigquery (or local)
    ingest_to_bigquery(raw_df)
    # 3) read data back (simulate BigQuery read)
    data = read_from_bigquery()
    # 4) train local model (or call Vertex AI)
    model = train_local_model(data)
    # 5) run optimization per record
    outputs = []
    for _, row in data.iterrows():
        rec = row.to_dict()
        out = optimize_tsr_for_record(model, rec)
        outputs.append(out)
    # 6) persist to BigQuery / local
    persist_optimizations(outputs)
    return outputs

# --------------------------
# 9) Flask API endpoint for on-demand optimization (Cloud Run)
# --------------------------
app = Flask(__name__)

@app.route("/optimize", methods=["POST"])
def api_optimize():
    payload = request.json
    if not payload:
        return jsonify({"error": "no payload"}), 400
    # expected payload contains fields like moisture_pct, calorific_value_kcal_kg, ash_content_pct, feed_rate_tph, current_tsr_pct
    model = train_local_model(read_from_bigquery(limit=200))  # lightweight retrain on latest for demo
    result = optimize_tsr_for_record(model, payload)
    # Persist single result
    persist_optimizations([result])
    return jsonify(result)

# --------------------------
# 10) Simple CLI / main entry
# --------------------------
def main():
    print("Starting TSR optimization demo pipeline (USE_GCP =", USE_GCP, ")")
    outputs = run_pipeline_demo(n_messages=100)
    print("Sample output (first 3):")
    for r in outputs[:3]:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
    # Optionally run Flask API (comment/uncomment as needed)
    # app.run(host="0.0.0.0", port=8080)
