Final Cement GenAI - Streamlit V1 (Production-ready baseline)
============================================================

What this repo contains (final functional baseline):
- /src : ingestion simulators, training scripts, GCP integration helpers (placeholders where needed)
- /ui  : Streamlit app (polished, functional)
- /models : training artifacts (generated when you run training)
- /pipelines : scripts to call Vertex batch prediction & training (placeholders)
- /infra : deployment notes (Terraform later)

IMPORTANT:
- This repo uses synthetic data (lightweight demo, a few hundred rows).
- For GCP integration (BigQuery, Vertex AI, GCS, Pub/Sub) you must have gcloud ADC credentials:
    gcloud auth application-default login
- Region configured: asia-south1 (Mumbai).
- To run locally:
    pip install -r requirements.txt
    streamlit run ui/streamlit_app.py
