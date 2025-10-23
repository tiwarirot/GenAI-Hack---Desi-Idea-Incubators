# GCP utility placeholders.
# This module provides helper functions to interact with BigQuery, GCS and Vertex AI.
# IMPORTANT: to actually run these you must have google-cloud-* packages installed and ADC credentials.
# The functions import google libraries within the function bodies to avoid import errors during repo creation.

import os, time

PROJECT = os.environ.get('GCP_PROJECT', 'your-gcp-project')
BUCKET = os.environ.get('GCS_BUCKET', 'your-data-bucket')
REGION = os.environ.get('GCP_REGION', 'asia-south1')
BQ_DATASET = os.environ.get('BQ_DATASET', 'cement_ops')

def write_config_to_bq(process_key, param_name, param_value):
    from google.cloud import bigquery
    client = bigquery.Client()
    table_id = f"{client.project}.{BQ_DATASET}.config"
    rows = [{{'process_key': process_key, 'param_name': param_name, 'param_value': str(param_value), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')}}]
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(errors)
    return True

def read_config_from_bq(process_key=None):
    from google.cloud import bigquery
    client = bigquery.Client()
    base = f"SELECT * FROM `{client.project}.{BQ_DATASET}.config`"
    q = base + (f" WHERE process_key='{process_key}'" if process_key else "")
    df = client.query(q).to_dataframe()
    return df

def upload_to_gcs(local_path, dest_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_path)
    return f'gs://{BUCKET}/{dest_path}'

def register_model_vertex(display_name, gcs_uri, serving_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest'):
    from google.cloud import aiplatform
    aiplatform.init(project=os.environ.get('GCP_PROJECT'), location=REGION)
    model = aiplatform.Model.upload(display_name=display_name, artifact_uri=gcs_uri, serving_container_image_uri=serving_image_uri)
    endpoint = model.deploy(machine_type='n1-standard-4', min_replica_count=1, max_replica_count=1)
    return {{'model': model.resource_name, 'endpoint': endpoint.resource_name}}
