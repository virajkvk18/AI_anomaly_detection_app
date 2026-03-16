from fastapi import FastAPI
import joblib
import numpy as np
import datetime
import json

app = FastAPI()

model = joblib.load("model/anomaly_model.pkl")

LOG_FILE = "monitoring/prediction_log.json"

@app.get("/")
def home():
    return {"message":"AI Anomaly Detection API Running"}

@app.get("/health")
def health():
    return {"status":"API running"}

@app.get("/model-info")
def model_info():
    return {
        "model": "Isolation Forest",
        "features": ["cpu_usage", "memory_usage", "disk_io"]
    }

@app.post("/predict")
def predict(cpu_usage: float, memory_usage: float, disk_io: float):

    data = np.array([[cpu_usage, memory_usage, disk_io]])

    prediction = model.predict(data)

    result = "ANOMALY" if prediction[0] == -1 else "NORMAL"

    # log prediction
    log_prediction(cpu_usage, memory_usage, disk_io, result)

    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_io": disk_io,
        "prediction": result
    }

def log_prediction(cpu_usage, memory_usage, disk_io, result):

    log_data = {
        "time": str(datetime.datetime.now()),
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_io": disk_io,
        "prediction": result
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(log_data)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

@app.get("/history")
def history(limit:int=10):

    with open(LOG_FILE,"r") as f:
        logs=json.load(f)

    return logs[-limit:]