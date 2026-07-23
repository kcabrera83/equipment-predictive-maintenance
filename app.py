"""Flask web server for equipment predictive maintenance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

_classifier = None
_rul_estimator = None


def get_models():
    global _classifier, _rul_estimator
    if _classifier is None:
        from equip_predict.models.failure_predictor import FailurePredictor
        from equip_predict.models.life_estimator import LifeEstimator
        _classifier = FailurePredictor.load("outputs/models/failure_predictor.pkl")
        _rul_estimator = LifeEstimator.load("outputs/models/life_estimator.pkl")
    return _classifier, _rul_estimator


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "equipment-predictive-maintenance"})


@app.route("/api/dashboard")
def api_dashboard():
    import json
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
    return jsonify({"report": report, "predictions": predictions})


@app.route("/api/predict_status", methods=["POST"])
def api_predict_status():
    try:
        import pandas as pd
        from equip_predict.utils.preprocessor import EquipPreprocessor
        data = request.get_json()
        row = {
            "vibration_x": float(data.get("vibration_x", 2.0)),
            "vibration_y": float(data.get("vibration_y", 1.5)),
            "vibration_z": float(data.get("vibration_z", 1.0)),
            "temperature": float(data.get("temperature", 65)),
            "pressure": float(data.get("pressure", 200)),
            "flow_rate": float(data.get("flow_rate", 150)),
            "motor_current": float(data.get("motor_current", 30)),
            "rpm": float(data.get("rpm", 1800)),
            "bearing_temp": float(data.get("bearing_temp", 75)),
            "oil_level": float(data.get("oil_level", 80)),
            "seal_pressure": float(data.get("seal_pressure", 50)),
            "suction_pressure": float(data.get("suction_pressure", 100)),
            "discharge_pressure": float(data.get("discharge_pressure", 200)),
            "power_consumption": float(data.get("power_consumption", 45)),
            "runtime_hours": float(data.get("runtime_hours", 5000)),
            "days_since_maintenance": float(data.get("days_since_maintenance", 30)),
        }
        df = pd.DataFrame([row])
        prep = EquipPreprocessor()
        prep.fit(df)
        features = prep.transform(df)
        classifier, _ = get_models()
        pred = classifier.predict(features)[0]
        proba = classifier.predict_proba(features)
        status_labels = {0: "normal", 1: "warning", 2: "preventive_maintenance",
                         3: "corrective_maintenance", 4: "failure_imminent"}
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
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/predict_rul", methods=["POST"])
def api_predict_rul():
    try:
        import pandas as pd
        from equip_predict.utils.preprocessor import EquipPreprocessor
        data = request.get_json()
        row = {
            "vibration_x": float(data.get("vibration_x", 2.0)),
            "vibration_y": float(data.get("vibration_y", 1.5)),
            "vibration_z": float(data.get("vibration_z", 1.0)),
            "temperature": float(data.get("temperature", 65)),
            "pressure": float(data.get("pressure", 200)),
            "flow_rate": float(data.get("flow_rate", 150)),
            "motor_current": float(data.get("motor_current", 30)),
            "rpm": float(data.get("rpm", 1800)),
            "bearing_temp": float(data.get("bearing_temp", 75)),
            "oil_level": float(data.get("oil_level", 80)),
            "seal_pressure": float(data.get("seal_pressure", 50)),
            "suction_pressure": float(data.get("suction_pressure", 100)),
            "discharge_pressure": float(data.get("discharge_pressure", 200)),
            "power_consumption": float(data.get("power_consumption", 45)),
            "runtime_hours": float(data.get("runtime_hours", 5000)),
            "days_since_maintenance": float(data.get("days_since_maintenance", 30)),
        }
        df = pd.DataFrame([row])
        prep = EquipPreprocessor()
        prep.fit(df)
        features = prep.transform(df)
        _, rul_model = get_models()
        rul = rul_model.predict(features)[0]
        return jsonify({
            "status": "success",
            "rul_days": round(float(rul), 0),
            "recommendation": "schedule_maintenance" if rul < 60 else "continue_monitoring",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/anomaly_check", methods=["POST"])
def api_anomaly_check():
    try:
        import pandas as pd
        from equip_predict.models.anomaly_detector import EquipmentAnomalyDetector
        data = request.get_json()
        df = pd.DataFrame([{
            "vibration_x": float(data.get("vibration_x", 2.0)),
            "vibration_y": float(data.get("vibration_y", 1.5)),
            "vibration_z": float(data.get("vibration_z", 1.0)),
            "temperature": float(data.get("temperature", 65)),
            "pressure": float(data.get("pressure", 200)),
            "motor_current": float(data.get("motor_current", 30)),
            "bearing_temp": float(data.get("bearing_temp", 75)),
            "power_consumption": float(data.get("power_consumption", 45)),
        }])
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
        return jsonify({
            "status": "success",
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(scores[0]), 4),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/docs")
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "Equipment Predictive Maintenance - Predictive Maintenance", "version": "1.0.0"},
        "paths": {
            "/": {"get": {"summary": "Main dashboard"}},
            "/api/health": {"get": {"summary": "Service health check"}},
            "/api/dashboard": {"get": {"summary": "Evaluation report and predictions"}},
            "/api/predict_status": {"post": {"summary": "Predict equipment operational status"}},
            "/api/predict_rul": {"post": {"summary": "Estimate remaining useful life (RUL) of equipment"}},
            "/api/anomaly_check": {"post": {"summary": "Detect anomalies in equipment sensors"}},
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
