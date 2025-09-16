import pandas as pd

def load_sensor_data(path='../data/sample_sensor_data.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

if __name__ == "__main__":
    df = load_sensor_data()
    print(df.head())
