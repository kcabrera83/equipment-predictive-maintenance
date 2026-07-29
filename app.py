
from fastapi import FastAPI, HTTPException
from fastapi.security import APIKeyHeader
import numpy as np
import joblib, os

API_KEY = os.getenv("API_KEY", "dev")
api_key_header = APIKeyHeader(name="X-API-Key")
app = FastAPI(title="Equipment Predictive Maintenance")

def load_pipeline(path: str) -> dict:
    return joblib.load(path)

def scale(data: dict, X: np.ndarray) -> np.ndarray:
    scaler = data.get("scaler")
    return scaler.transform(X) if scaler else X

def run(data: dict, X: np.ndarray) -> float:
    return float(data["model"].predict(X)[0])

PIPELINES = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        PIPELINES[f.replace(".pkl", "")] = load_pipeline(os.path.join("outputs/models", f))

@app.get("/")
def index():
    return {"pipeline": "Equipment Predictive Maintenance", "available": list(PIPELINES.keys())}

@app.post("/pipeline/{name}")
def execute(name: str, body: dict, key: str = ""):
    if key != API_KEY:
        raise HTTPException(401, "Unauthorized")
    pipe = PIPELINES.get(name)
    if not pipe:
        raise HTTPException(404, f"Pipeline {name} not found")
    feats = pipe.get("feature_names", list(body.keys()))
    X = np.array([body.get(f, 0) for f in feats]).reshape(1, -1)
    X = scale(pipe, X)
    return {"result": run(pipe, X), "model": name}
