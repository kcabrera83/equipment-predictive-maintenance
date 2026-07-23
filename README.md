# Equipment Predictive Maintenance

Predictive maintenance system for oil & gas equipment based on Machine Learning. Classifies equipment status (normal, warning, preventive/corrective maintenance, imminent failure), estimates remaining useful life (RUL), and detects sensor anomalies.

---

## Architecture

```
equipment-predictive-maintenance/
├── equip_predict/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic sensor data generator
│   ├── models/
│   │   ├── failure_predictor.py   # Status classification (5 classes)
│   │   ├── life_estimator.py      # RUL regression (remaining days)
│   │   └── anomaly_detector.py    # Anomaly detection (Isolation Forest)
│   └── utils/
│       ├── preprocessor.py        # Sensor feature engineering
│       ├── visualizer.py          # Trend and evaluation charts
│       └── metrics.py             # Accuracy, F1, MAE, R2, confusion matrix
├── scripts/
│   ├── train.py                   # Full training
│   ├── predict.py                 # Equipment prediction
│   └── evaluate.py                # Evaluation with CV
├── templates/
│   └── index.html                 # Interactive web dashboard
├── app.py                         # Flask server (5 endpoints)
├── test_api.py                    # Automated tests
├── requirements.txt
├── .github/workflows/test.yml     # CI/CD
└── README.md
```

---

## ML Models

### Status Classification

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| GradientBoosting | 0.99+ | 0.99+ |
| ExtraTrees | 0.99+ | 0.99+ |
| RandomForest | 0.98+ | 0.98+ |
| MLP | 0.95+ | 0.95+ |
| LogisticRegression | 0.85+ | 0.85+ |

### RUL Estimation (Remaining Useful Life)

| Model | R2 | MAE (days) |
|-------|-----|------------|
| ExtraTrees | 0.99+ | < 15 |
| GradientBoosting | 0.99+ | < 15 |
| RandomForest | 0.98+ | < 20 |
| MLP | 0.95+ | < 25 |
| Ridge | 0.90+ | < 30 |

---

## Supported Equipment

| Type | Vibration | Temperature | Pressure | Useful Life |
|------|-----------|-------------|----------|-------------|
| Centrifugal Pump | 0.5-8 mm/s | 40-95 C | 50-400 psi | 180-720 days |
| Reciprocating Pump | 2-15 mm/s | 35-85 C | 100-1000 psi | 240-900 days |
| Compressor | 1-12 mm/s | 50-150 C | 200-2000 psi | 365-1460 days |
| Separator | 0.3-4 mm/s | 30-70 C | 50-300 psi | 500-1800 days |

---

## Sensors (16 features)

- vibration_x, vibration_y, vibration_z — 3-axis vibration (mm/s)
- temperature — Equipment temperature (C)
- pressure — Operating pressure (psi)
- flow_rate — Volumetric flow (m3/h)
- motor_current — Motor current (A)
- rpm — Rotation speed
- bearing_temp — Bearing temperature (C)
- oil_level — Oil level (%)
- seal_pressure — Seal pressure (psi)
- suction_pressure — Suction pressure (psi)
- discharge_pressure — Discharge pressure (psi)
- power_consumption — Power consumption (kW)
- runtime_hours — Operating hours
- days_since_maintenance — Days since last maintenance

### Derived Features (8)

- vibration_total, vibration_trend, temp_pressure_ratio, efficiency
- power_per_flow, seal_pressure_ratio, discharge_suction_diff, bearing_temp_diff

---

## Equipment States

| Code | State | Description |
|------|-------|-------------|
| 0 | Normal | Operation within parameters |
| 1 | Warning | Elevated values, monitor |
| 2 | Preventive Maintenance | Scheduled maintenance recommended |
| 3 | Corrective Maintenance | Corrective maintenance needed |
| 4 | Failure Imminent | Imminent failure, stop equipment |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service status |
| `/api/dashboard` | GET | Metrics + predictions |
| `/api/predict_status` | POST | Status classification (16 inputs) |
| `/api/predict_rul` | POST | RUL estimation in days |
| `/api/anomaly_check` | POST | Sensor anomaly detection |

---

## Web Dashboard

Flask + Chart.js — 4 sections:

1. **Dashboard** — model metrics and equipment table
2. **Status Prediction** — 16-sensor form for classification
3. **RUL Estimation** — remaining useful life estimation with recommendation
4. **Anomaly Detection** — anomaly verification with score

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
python scripts/evaluate.py
python app.py
# Open http://localhost:5000
```

---

## CI/CD

GitHub Actions runs on every push:

1. Model training
2. Predictions
3. Full evaluation
4. API tests (5 endpoints)

---

Elaborado por Ing. Kelvin Cabrera
