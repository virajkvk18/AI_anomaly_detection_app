import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/system_metrics.csv")

X = df[["cpu_usage","memory_usage","disk_io"]]

model = IsolationForest(contamination=0.05)

model.fit(X)

joblib.dump(model,"model/anomaly_model.pkl")

print("Model trained on real system data")