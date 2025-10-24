import os, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
try:
    import lightgbm as lgb
except Exception:
    lgb = None

from src.data_generator import gen_raw_grinding, gen_clinker, gen_quality, gen_altfuel, gen_cross
from src.gcp_utils import upload_to_gcs, register_model_vertex

MODELDIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELDIR, exist_ok=True)

def train_raw_grinding(n=500):
    df = gen_raw_grinding(n)
    X = df[['raw_material_variability','grinding_efficiency']]
    y = df['energy_consumption']
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X,y)
    path = os.path.join(MODELDIR, 'raw_grinding.joblib')
    joblib.dump(m, path)
    return path

def train_clinker(n=500):
    df = gen_clinker(n)
    X = df[['kiln_temp','feed_rate','oxygen_level']]
    y = df['energy_use']
    if lgb:
        m = lgb.LGBMRegressor(n_estimators=200)
        m.fit(X,y)
    else:
        m = RandomForestRegressor(n_estimators=100)
        m.fit(X,y)
    path = os.path.join(MODELDIR, 'clinker.joblib')
    joblib.dump(m, path)
    return path

def train_quality(n=500):
    df = gen_quality(n)
    X = df[['si_o2','moisture','blaine']]
    y = df['compressive_strength']
    m = RandomForestRegressor(n_estimators=100)
    m.fit(X,y)
    path = os.path.join(MODELDIR, 'quality.joblib')
    joblib.dump(m, path)
    return path

def train_altfuel(n=500):
    df = gen_altfuel(n)
    X = df[['fuel_calorific','rfd_share','tsr']]
    y = df['energy_consumption']
    if lgb:
        m = lgb.LGBMRegressor(n_estimators=200)
        m.fit(X,y)
    else:
        m = RandomForestRegressor(n_estimators=100)
        m.fit(X,y)
    path = os.path.join(MODELDIR, 'altfuel.joblib')
    joblib.dump(m, path)
    return path

def train_cross(n=500):
    df = gen_cross(n)
    X = df[['raw_material_variability','grinding_efficiency','kiln_temp','feed_rate','oxygen_level','tsr','fuel_calorific']]
    if 'energy_consumption' in df.columns:
        y = df['energy_consumption']
    elif 'energy_use' in df.columns:
        y = df['energy_use']
    else:
        y = np.random.normal(120,10,len(df))
    m = RandomForestRegressor(n_estimators=100)
    m.fit(X,y)
    path = os.path.join(MODELDIR, 'cross.joblib')
    joblib.dump(m, path)
    return path

def upload_and_register(local_path, display_name):
    dest = f'models/{os.path.basename(local_path)}'
    gcs_uri = upload_to_gcs(local_path, dest)
    res = register_model_vertex(display_name, gcs_uri)
    return res
