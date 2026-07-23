# API Documentation - Equipment Predictive Maintenance

## Base URL
```
http://localhost:5003
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with 4 sections (Dashboard, Status Prediction, RUL Estimation, Anomaly Detection).

---

### GET /api/health
Service health check.

**Response (200):**
```json
{"status": "ok", "service": "equipment-predictive-maintenance"}
```

---

### GET /api/dashboard
Evaluation report and sample predictions.

**Response (200):**
```json
{
  "report": {
    "best_classifier": "GradientBoosting",
    "classifier_metrics": {
      "accuracy": 0.99,
      "f1_macro": 0.99
    }
  },
  "predictions": [...]
}
```

---

### POST /api/predict_status
Predict equipment operational status from 16 sensor readings.

**Request:**
```json
{
  "vibration_x": 2.0,
  "vibration_y": 1.5,
  "vibration_z": 1.0,
  "temperature": 65,
  "pressure": 200,
  "flow_rate": 150,
  "motor_current": 30,
  "rpm": 1800,
  "bearing_temp": 75,
  "oil_level": 80,
  "seal_pressure": 50,
  "suction_pressure": 100,
  "discharge_pressure": 200,
  "power_consumption": 45,
  "runtime_hours": 5000,
  "days_since_maintenance": 30
}
```

**Response (200):**
```json
{
  "status": "success",
  "predicted_status": "normal",
  "status_code": 0,
  "probabilities": {
    "normal": 0.9200,
    "warning": 0.0500,
    "preventive_maintenance": 0.0200,
    "corrective_maintenance": 0.0080,
    "failure_imminent": 0.0020
  }
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### POST /api/predict_rul
Estimate Remaining Useful Life (RUL) of equipment in days.

**Request:**
```json
{
  "vibration_x": 2.0,
  "vibration_y": 1.5,
  "vibration_z": 1.0,
  "temperature": 65,
  "pressure": 200,
  "flow_rate": 150,
  "motor_current": 30,
  "rpm": 1800,
  "bearing_temp": 75,
  "oil_level": 80,
  "seal_pressure": 50,
  "suction_pressure": 100,
  "discharge_pressure": 200,
  "power_consumption": 45,
  "runtime_hours": 5000,
  "days_since_maintenance": 30
}
```

**Response (200):**
```json
{
  "status": "success",
  "rul_days": 180,
  "recommendation": "continue_monitoring"
}
```

**Recommendation Logic:**
- `rul_days < 60` -> `"schedule_maintenance"`
- `rul_days >= 60` -> `"continue_monitoring"`

---

### POST /api/anomaly_check
Detect anomalies in equipment sensor readings.

**Request:**
```json
{
  "vibration_x": 2.0,
  "vibration_y": 1.5,
  "vibration_z": 1.0,
  "temperature": 65,
  "pressure": 200,
  "motor_current": 30,
  "bearing_temp": 75,
  "power_consumption": 45
}
```

**Response (200):**
```json
{
  "status": "success",
  "is_anomaly": false,
  "anomaly_score": -0.0350
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - invalid input or processing error |
| 500 | Internal server error |
