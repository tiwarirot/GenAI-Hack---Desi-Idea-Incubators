import pandas as pd
import numpy as np

def gen_raw_grinding(n=300, seed=1):
    np.random.seed(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range(start='2025-01-01', periods=n, freq='T'),
        "raw_material_variability": np.clip(np.random.normal(0.2,0.05,n), 0.01, 0.6),
        "grinding_efficiency": np.clip(np.random.normal(0.8,0.06,n), 0.5, 1.0),
        "energy_consumption": np.random.normal(9.6,0.4,n)
    })
    return df

def gen_clinker(n=300, seed=2):
    np.random.seed(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range(start='2025-01-01', periods=n, freq='T'),
        "kiln_temp": np.random.normal(1450,12,n),
        "feed_rate": np.random.normal(300,18,n),
        "oxygen_level": np.random.normal(3.5,0.35,n),
        "energy_use": np.random.normal(150,7,n)
    })
    return df

def gen_quality(n=300, seed=3):
    np.random.seed(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range(start='2025-01-01', periods=n, freq='T'),
        "si_o2": np.random.normal(21,1.2,n),
        "moisture": np.random.normal(4.0,0.6,n),
        "blaine": np.random.normal(330,18,n),
        "compressive_strength": np.random.normal(40,3,n)
    })
    return df

def gen_altfuel(n=300, seed=4):
    np.random.seed(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range(start='2025-01-01', periods=n, freq='T'),
        "fuel_calorific": np.random.normal(3500,250,n),
        "rfd_share": np.random.uniform(0.0,0.45,n),
        "tsr": np.random.uniform(10,40,n),
        "energy_consumption": np.random.normal(140,9,n)
    })
    return df

def gen_cross(n=300, seed=5):
    rg = gen_raw_grinding(n, seed)
    cl = gen_clinker(n, seed+1)
    af = gen_altfuel(n, seed+2)
    df = pd.concat([rg.reset_index(drop=True), cl[['kiln_temp','feed_rate','oxygen_level']].reset_index(drop=True), af[['tsr','fuel_calorific']].reset_index(drop=True)], axis=1)
    return df
