# Equipment Predictive Maintenance

Sistema de mantenimiento predictivo para equipos de oil & gas basado en Machine Learning. Clasifica el estado de equipos (normal, warning, mantenimiento preventivo/correctivo, fallo inminente), estima la vida util restante (RUL) y detecta anomalias en sensores.

---

## Arquitectura

```
equipment-predictive-maintenance/
├── equip_predict/
│   ├── __init__.py
│   ├── data_generator.py          # Generador de datos sinteticos de sensores
│   ├── models/
│   │   ├── failure_predictor.py   # Clasificacion de estado (5 clases)
│   │   ├── life_estimator.py      # Regresion de RUL (dias restantes)
│   │   └── anomaly_detector.py    # Deteccion de anomalias (Isolation Forest)
│   └── utils/
│       ├── preprocessor.py        # Feature engineering de sensores
│       ├── visualizer.py          # Graficos de tendencias y evaluacion
│       └── metrics.py             # Accuracy, F1, MAE, R2, confusion matrix
├── scripts/
│   ├── train.py                   # Entrenamiento completo
│   ├── predict.py                 # Prediccion por equipo
│   └── evaluate.py                # Evaluacion con CV
├── templates/
│   └── index.html                 # Dashboard web interactivo
├── app.py                         # Servidor Flask (5 endpoints)
├── test_api.py                    # Tests automatizados
├── requirements.txt
├── .github/workflows/test.yml     # CI/CD
└── README.md
```

---

## Modelos ML

### Clasificacion de Estado

| Modelo | Accuracy | F1 Macro |
|--------|----------|----------|
| GradientBoosting | 0.99+ | 0.99+ |
| ExtraTrees | 0.99+ | 0.99+ |
| RandomForest | 0.98+ | 0.98+ |
| MLP | 0.95+ | 0.95+ |
| LogisticRegression | 0.85+ | 0.85+ |

### Estimacion de RUL (Remaining Useful Life)

| Modelo | R2 | MAE (days) |
|--------|-----|------------|
| ExtraTrees | 0.99+ | < 15 |
| GradientBoosting | 0.99+ | < 15 |
| RandomForest | 0.98+ | < 20 |
| MLP | 0.95+ | < 25 |
| Ridge | 0.90+ | < 30 |

---

## Equipos Soportados

| Tipo | Vibracion | Temperatura | Presion | Vida Util |
|------|-----------|-------------|---------|-----------|
| Centrifugal Pump | 0.5-8 mm/s | 40-95 C | 50-400 psi | 180-720 days |
| Reciprocating Pump | 2-15 mm/s | 35-85 C | 100-1000 psi | 240-900 days |
| Compressor | 1-12 mm/s | 50-150 C | 200-2000 psi | 365-1460 days |
| Separator | 0.3-4 mm/s | 30-70 C | 50-300 psi | 500-1800 days |

---

## Sensores (16 features)

- vibration_x, vibration_y, vibration_z — Vibracion 3-eje (mm/s)
- temperature — Temperatura del equipo (C)
- pressure — Presion de operacion (psi)
- flow_rate — Flujo volumetrico (m3/h)
- motor_current — Corriente del motor (A)
- rpm — Velocidad de rotacion
- bearing_temp — Temperatura del rodamiento (C)
- oil_level — Nivel de aceite (%)
- seal_pressure — Presion del sello (psi)
- suction_pressure — Presion de succion (psi)
- discharge_pressure — Presion de descarga (psi)
- power_consumption — Consumo electrico (kW)
- runtime_hours — Horas de operacion
- days_since_maintenance — Dias desde ultimo mantenimiento

### Features Derivadas (8)

- vibration_total, vibration_trend, temp_pressure_ratio, efficiency
- power_per_flow, seal_pressure_ratio, discharge_suction_diff, bearing_temp_diff

---

## Estados del Equipo

| Codigo | Estado | Descripcion |
|--------|--------|-------------|
| 0 | Normal | Operacion dentro de parametros |
| 1 | Warning | Valores elevados, monitorear |
| 2 | Preventive Maintenance | Mantenimiento programado recomendado |
| 3 | Corrective Maintenance | Mantenimiento correctivo necesario |
| 4 | Failure Imminent | Fallo inminente, detener equipo |

---

## API Endpoints

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/api/health` | GET | Estado del servicio |
| `/api/dashboard` | GET | Metricas + predicciones |
| `/api/predict_status` | POST | Clasificacion de estado (16 inputs) |
| `/api/predict_rul` | POST | Estimacion de RUL en dias |
| `/api/anomaly_check` | POST | Deteccion de anomalias en sensores |

---

## Dashboard Web

Flask + Chart.js — 4 secciones:

1. **Dashboard** — metricas del modelo y tabla de equipos
2. **Status Prediction** — formulario de 16 sensores para clasificacion
3. **RUL Estimation** — estimacion de vida util restante con recomendacion
4. **Anomaly Detection** — verificacion de anomalias con score

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
python scripts/evaluate.py
python app.py
# Abrir http://localhost:5000
```

---

## CI/CD

GitHub Actions ejecuta en cada push:

1. Entrenamiento de modelos
2. Predicciones
3. Evaluacion completa
4. Tests de API (5 endpoints)

---

Elaborado por Ing. Kelvin Cabrera
