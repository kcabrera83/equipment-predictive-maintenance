# User Guide - Equipment Predictive Maintenance

## Overview
Predictive maintenance system for oil & gas equipment. Classifies equipment status (normal, warning, preventive/corrective maintenance, imminent failure), estimates remaining useful life (RUL), and detects sensor anomalies.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/equipment-predictive-maintenance.git
cd equipment-predictive-maintenance
pip install -r requirements.txt
```

### Training Models
```bash
python scripts/train.py
```

### Running Predictions
```bash
python scripts/predict.py
```

### Full Evaluation
```bash
python scripts/evaluate.py
```

### Starting the Server
```bash
python app.py
```
Open http://localhost:5003 in your browser.

## Dashboard Features

### 1. Dashboard Section
- Model comparison table for classifiers and RUL estimators
- Best model metrics (Accuracy, F1, R2, MAE)
- Sample equipment status predictions

### 2. Status Prediction Section
- 16-sensor input form for equipment status classification
- 5 possible status outputs with probability breakdown
- Color-coded severity indicators

### 3. RUL Estimation Section
- Same 16-sensor inputs for remaining useful life prediction
- Output: estimated days remaining
- Automatic maintenance recommendation

### 4. Anomaly Detection Section
- 8-field anomaly verification form
- Isolation Forest-based anomaly scoring
- Binary anomaly classification

## Equipment States
| Code | State | Description |
|------|-------|-------------|
| 0 | Normal | Operation within normal parameters |
| 1 | Warning | Elevated values, increased monitoring recommended |
| 2 | Preventive Maintenance | Scheduled maintenance recommended |
| 3 | Corrective Maintenance | Corrective maintenance needed |
| 4 | Failure Imminent | Stop equipment immediately |

## Supported Equipment Types
| Type | Vibration (mm/s) | Temperature (C) | Pressure (psi) |
|------|-------------------|-----------------|----------------|
| Centrifugal Pump | 0.5-8 | 40-95 | 50-400 |
| Reciprocating Pump | 2-15 | 35-85 | 100-1000 |
| Compressor | 1-12 | 50-150 | 200-2000 |
| Separator | 0.3-4 | 30-70 | 50-300 |

## API Usage

### Using curl
```bash
# Predict equipment status
curl -X POST http://localhost:5003/api/predict_status \
  -H "Content-Type: application/json" \
  -d '{"vibration_x": 2.0, "vibration_y": 1.5, "vibration_z": 1.0, "temperature": 65, "pressure": 200, "motor_current": 30, "bearing_temp": 75, "power_consumption": 45, "rpm": 1800, "runtime_hours": 5000}'

# Estimate remaining useful life
curl -X POST http://localhost:5003/api/predict_rul \
  -H "Content-Type: application/json" \
  -d '{"vibration_x": 2.0, "vibration_y": 1.5, "vibration_z": 1.0, "temperature": 65, "pressure": 200, "motor_current": 30, "bearing_temp": 75, "power_consumption": 45}'

# Check for anomalies
curl -X POST http://localhost:5003/api/anomaly_check \
  -H "Content-Type: application/json" \
  -d '{"vibration_x": 2.0, "vibration_y": 1.5, "vibration_z": 1.0, "temperature": 65, "pressure": 200, "motor_current": 30, "bearing_temp": 75, "power_consumption": 45}'
```

### Using Python
```python
import requests

# Predict equipment status
response = requests.post("http://localhost:5003/api/predict_status", json={
    "vibration_x": 2.0, "vibration_y": 1.5, "vibration_z": 1.0,
    "temperature": 65, "pressure": 200, "motor_current": 30,
    "bearing_temp": 75, "power_consumption": 45, "rpm": 1800
})
result = response.json()
print(f"Status: {result['predicted_status']}")
print(f"Probabilities: {result['probabilities']}")

# Estimate RUL
response = requests.post("http://localhost:5003/api/predict_rul", json={
    "vibration_x": 2.0, "vibration_y": 1.5, "vibration_z": 1.0,
    "temperature": 65, "pressure": 200, "motor_current": 30,
    "bearing_temp": 75, "power_consumption": 45
})
rul = response.json()
print(f"RUL: {rul['rul_days']} days - {rul['recommendation']}")
```

## Sensor Features (16 raw + 8 derived)
### Raw Sensors
vibration_x/y/z, temperature, pressure, flow_rate, motor_current, rpm, bearing_temp, oil_level, seal_pressure, suction_pressure, discharge_pressure, power_consumption, runtime_hours, days_since_maintenance

### Derived Features
vibration_total, vibration_trend, temp_pressure_ratio, efficiency, power_per_flow, seal_pressure_ratio, discharge_suction_diff, bearing_temp_diff
