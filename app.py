"""FastAPI web server for equipment predictive maintenance."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

app = FastAPI(
    title="Equipment Predictive Maintenance",
    description="Equipment status prediction, remaining useful life estimation, and anomaly detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    from equip_predict.models.failure_predictor import FailurePredictor
    from equip_predict.models.life_estimator import LifeEstimator
    try:
        models["classifier"] = FailurePredictor.load("outputs/models/failure_predictor.pkl")
        models["rul_estimator"] = LifeEstimator.load("outputs/models/life_estimator.pkl")
    except Exception as e:
        print(f"  Error loading models: {e}")


class SensorInput(BaseModel):
    vibration_x: float = 2.0
    vibration_y: float = 1.5
    vibration_z: float = 1.0
    temperature: float = 65.0
    pressure: float = 200.0
    flow_rate: float = 150.0
    motor_current: float = 30.0
    rpm: float = 1800.0
    bearing_temp: float = 75.0
    oil_level: float = 80.0
    seal_pressure: float = 50.0
    suction_pressure: float = 100.0
    discharge_pressure: float = 200.0
    power_consumption: float = 45.0
    runtime_hours: float = 5000.0
    days_since_maintenance: float = 30.0


class AnomalyInput(BaseModel):
    vibration_x: float = 2.0
    vibration_y: float = 1.5
    vibration_z: float = 1.0
    temperature: float = 65.0
    pressure: float = 200.0
    motor_current: float = 30.0
    bearing_temp: float = 75.0
    power_consumption: float = 45.0


def _to_df(data):
    return pd.DataFrame([{k: float(v) for k, v in data.items()}])


def _preprocess(df):
    from equip_predict.utils.preprocessor import EquipPreprocessor
    prep = EquipPreprocessor()
    prep.fit(df)
    return prep.transform(df)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "equipment-predictive-maintenance"}


@app.get("/api/dashboard")
async def api_dashboard():
    report_path = Path("outputs/evaluation_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {"best_classifier": "N/A", "classifier_metrics": {}}
    predictions_path = Path("outputs/predictions.json")
    if predictions_path.exists():
        with open(predictions_path) as f:
            predictions = json.load(f)
    else:
        predictions = []
    return {"report": report, "predictions": predictions}


@app.post("/api/predict_status")
async def api_predict_status(request: SensorInput):
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    try:
        df = _to_df(request.model_dump())
        features = _preprocess(df)
        pred = models["classifier"].predict(features)[0]
        proba = models["classifier"].predict_proba(features)
        status_labels = {
            0: "normal", 1: "warning", 2: "preventive_maintenance",
            3: "corrective_maintenance", 4: "failure_imminent",
        }
        result = {
            "status": "success",
            "predicted_status": status_labels.get(int(pred), str(pred)),
            "status_code": int(pred),
        }
        if proba is not None:
            result["probabilities"] = {
                status_labels.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(proba[0])
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict_rul")
async def api_predict_rul(request: SensorInput):
    if "rul_estimator" not in models:
        raise HTTPException(status_code=503, detail="RUL estimator not loaded")
    try:
        df = _to_df(request.model_dump())
        features = _preprocess(df)
        rul = models["rul_estimator"].predict(features)[0]
        return {
            "status": "success",
            "rul_days": round(float(rul), 0),
            "recommendation": "schedule_maintenance" if rul < 60 else "continue_monitoring",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/anomaly_check")
async def api_anomaly_check(request: AnomalyInput):
    try:
        from equip_predict.models.anomaly_detector import EquipmentAnomalyDetector
        df = _to_df(request.model_dump())
        rng = np.random.default_rng(42)
        normal_data = pd.DataFrame({
            "vibration_x": rng.uniform(0.5, 5, 100),
            "vibration_y": rng.uniform(0.3, 4, 100),
            "vibration_z": rng.uniform(0.2, 3, 100),
            "temperature": rng.uniform(40, 90, 100),
            "pressure": rng.uniform(50, 400, 100),
            "motor_current": rng.uniform(10, 80, 100),
            "bearing_temp": rng.uniform(50, 110, 100),
            "power_consumption": rng.uniform(20, 80, 100),
        })
        detector = EquipmentAnomalyDetector()
        detector.fit(normal_data)
        predictions, scores = detector.predict(df)
        is_anomaly = predictions[0] == -1
        return {
            "status": "success",
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(scores[0]), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)

