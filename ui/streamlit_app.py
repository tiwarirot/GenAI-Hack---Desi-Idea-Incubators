import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import streamlit as st, os, time, joblib
from pathlib import Path
repo_root = Path(__file__).parents[1].resolve()
import sys
sys.path.append(str(repo_root / 'src'))

from data_generator import gen_raw_grinding, gen_clinker, gen_quality, gen_altfuel, gen_cross
from trainers import train_raw_grinding, train_clinker, train_quality, train_altfuel, train_cross, upload_and_register
from ingest_simulator import write_all as ingest_write_all
from gcp_utils import write_config_to_bq, read_config_from_bq

st.set_page_config(layout='wide', page_title='GenAI Cement - Final Streamlit')
st.title('GenAI Cement — Operator Dashboard')

st.sidebar.title('Navigation')
menu = st.sidebar.radio('Go to', ['Overview','Raw & Grinding','Clinker','Quality','Alt Fuel','Cross','Config','Train'])

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def load_csv(name):
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        import pandas as pd
        return pd.read_csv(p)
    else:
        return None

if menu == 'Overview':
    st.header('Overview')
    st.write('Lightweight synthetic data demo.')
    if st.button('Ingest synthetic data (write CSVs)'):
        ingest_write_all()
        st.success('Synthetic data written to data/ directory.')

if menu == 'Raw & Grinding':
    st.header('Optimize Raw Materials & Grinding')
    df = load_csv('raw_grinding.csv')
    if df is None:
        st.warning('No data found; run ingestion from Overview.')
    else:
        st.dataframe(df.head(8))
    v = st.slider('Raw Material Variability', 0.05, 0.5, 0.2)
    e = st.slider('Grinding Efficiency', 0.5, 1.0, 0.85)
    if st.button('Predict (local)'):
        with st.spinner('Training small local model and predicting...'):
            path = train_raw_grinding(n=300)
            m = joblib.load(path)
            pred = m.predict([[v,e]])[0]
            st.success(f'Predicted energy consumption: {pred:.2f} kWh/t')
    if st.button('Trigger Vertex Batch (placeholder)'):
        st.info('This would submit a Vertex AI batch prediction job (placeholder).')

if menu == 'Clinker':
    st.header('Balance Clinkerization Parameters')
    df = load_csv('clinker.csv')
    if df is None:
        st.warning('No data found; run ingestion from Overview.')
    else:
        st.dataframe(df.head(8))
    temp = st.slider('Kiln Temp (°C)', 1400,1500,1450)
    feed = st.slider('Feed Rate (tph)', 250,350,300)
    oxy = st.slider('Oxygen Level (%)', 2.0,5.0,3.5)
    if st.button('Predict (local)'):
        with st.spinner('Training local clinker model...'):
            path = train_clinker(n=300)
            m = joblib.load(path)
            pred = m.predict([[temp,feed,oxy]])[0]
            st.success(f'Predicted energy use: {pred:.2f}')

if menu == 'Quality':
    st.header('Ensure Quality Consistency')
    df = load_csv('quality.csv')
    if df is None:
        st.warning('No data found; run ingestion from Overview.')
    else:
        st.dataframe(df.head(8))
    si = st.number_input('SiO2 %', 10.0,30.0,21.0)
    moist = st.number_input('Moisture %', 0.1,10.0,4.0)
    bl = st.number_input('Blaine', 200,500,330)
    if st.button('Predict (local)'):
        with st.spinner('Training quality model...'):
            path = train_quality(n=300)
            m = joblib.load(path)
            pred = m.predict([[si,moist,bl]])[0]
            st.success(f'Predicted compressive strength: {pred:.2f} MPa')

if menu == 'Alt Fuel':
    st.header('Maximize Alternative Fuel Use (TSR)')
    df = load_csv('altfuel.csv')
    if df is None:
        st.warning('No data found; run ingestion from Overview.')
    else:
        st.dataframe(df.head(8))
    fuel = st.number_input('Fuel Calorific (kcal/kg)', 1000,6000,3500)
    rfd = st.slider('RDF share', 0.0,0.8,0.3)
    tsr = st.slider('Current TSR %', 0.0,60.0,32.0)
    if st.button('Predict (local)'):
        with st.spinner('Training altfuel model...'):
            path = train_altfuel(n=300)
            m = joblib.load(path)
            pred = m.predict([[fuel,rfd,tsr]])[0]
            st.success(f'Predicted energy consumption: {pred:.2f}')

if menu == 'Cross':
    st.header('Strategic Cross-Process Optimization')
    df = load_csv('cross.csv')
    if df is None:
        st.warning('No data found; run ingestion from Overview.')
    else:
        st.dataframe(df.head(6))
    if st.button('Predict (local)'):
        with st.spinner('Training cross model...'):
            path = train_cross(n=300)
            m = joblib.load(path)
            row = df.tail(1)
            X = row[['raw_material_variability','grinding_efficiency','kiln_temp','feed_rate','oxygen_level','tsr','fuel_calorific']].values[0].tolist()
            pred = m.predict([X])[0]
            st.success(f'Cross predicted energy: {pred:.2f}')

if menu == 'Config':
    st.header('Edit and persist configuration (BigQuery)')
    st.write('Requires GCP ADC credentials and dataset.table config exists.')
    proc = st.selectbox('Process', ['RM','Clinker','Quality','Fuel','Cross'])
    pname = st.text_input('Parameter name', 'tsr_threshold')
    pval = st.text_input('Parameter value', '0.2')
    if st.button('Save to BigQuery'):
        try:
            write_config_to_bq(proc, pname, pval)
            st.success('Saved config to BigQuery')
        except Exception as ex:
            st.error('Failed: ' + str(ex))
    if st.button('Read Configs'):
        try:
            dfc = read_config_from_bq(None)
            st.dataframe(dfc)
        except Exception as ex:
            st.error('Failed: ' + str(ex))

if menu == 'Train':
    st.header('Train and optionally register model')
    if st.button('Train All Local Models'):
        with st.spinner('Training all...'):
            train_raw_grinding(); train_clinker(); train_quality(); train_altfuel(); train_cross()
            st.success('All local models trained and saved to /models')
    if st.button('Upload & register Raw Grinding model to Vertex (requires GCP)'):
        try:
            path = os.path.join(os.path.dirname(__file__), '..', 'models', 'raw_grinding.joblib')
            res = upload_and_register(path, 'raw-grinding-demo')
            st.write(res)
        except Exception as ex:
            st.error('Register failed: ' + str(ex))
