# Architecture - Equipment Predictive Maintenance

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5003       |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v-------+  +------v--------+  +------v--------+
| Failure         |  | Life          |  | Anomaly       |
| Predictor       |  | Estimator     |  | Detector      |
| (5-class CLS)   |  | (RUL REG)     |  | (Iso Forest)  |
+---------+-------+  +------+--------+  +------+--------+
          |                  |                  |
+---------v------------------v------------------v-------+
|              EquipPreprocessor                         |
|       (Feature Engineering + Scaling)                  |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (50 units x 120d)  |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic sensor data generator (`EquipDataGenerator`) producing 50 equipment units with 120 days of readings
- **Equipment Types**: Centrifugal Pump, Reciprocating Pump, Compressor, Separator
- **Sensor Data**: 16 raw features + 8 derived features (24 total)
- **Preprocessing**: Feature engineering, standard scaling, train/test split

### Model Layer

#### Failure Predictor (Status Classification)
- **Algorithm**: GradientBoosting (best) among ExtraTrees, RandomForest, MLP, LogisticRegression
- **Target**: Equipment status (5 classes: normal, warning, preventive_maintenance, corrective_maintenance, failure_imminent)
- **Metrics**: Accuracy > 0.99, F1 Macro > 0.99
- **Output**: Class label + probability distribution across all 5 states

#### Life Estimator (RUL Regression)
- **Algorithm**: ExtraTrees (best) among GradientBoosting, RandomForest, MLP, Ridge
- **Target**: Remaining Useful Life in days
- **Metrics**: R2 > 0.99, MAE < 15 days
- **Output**: Predicted days of useful life remaining

#### Anomaly Detector
- **Algorithm**: Isolation Forest
- **Input**: 8 key sensor readings (vibration, temperature, pressure, motor_current, bearing_temp, power_consumption)
- **Purpose**: Detect abnormal sensor patterns

### API Layer
- **Framework**: Flask
- **Endpoints**: 6 REST endpoints (health, dashboard, predict_status, predict_rul, anomaly_check, docs)
- **Model Loading**: Lazy loading on first request

### Dashboard Layer
- **Frontend**: Flask + Chart.js
- **Sections**: 4 interactive panels (Dashboard, Status Prediction, RUL Estimation, Anomaly Detection)
- **Charts**: Model comparison, confusion matrix, RUL scatter plots

## Data Flow

1. **Input**: 16 sensor readings from equipment
2. **Feature Engineering**: `EquipPreprocessor` computes 8 derived features (vibration_total, efficiency, ratios, etc.)
3. **Status Classification**: FailurePredictor classifies into 5 equipment states with probabilities
4. **RUL Estimation**: LifeEstimator predicts remaining useful life in days
5. **Anomaly Detection**: Isolation Forest scores sensor readings for anomalies
6. **Dashboard**: All results displayed in interactive panels

## Training Pipeline
1. Generate synthetic sensor data (50 units, 120 days)
2. Visualize failure distribution and RUL distribution
3. Prepare classification dataset (24 features, 5 classes)
4. Train 5 classifiers, select best by F1-macro
5. Prepare RUL regression dataset
6. Train 5 regressors, select best by R2
7. Save best models to `outputs/models/`
8. Generate evaluation plots, confusion matrix, RUL predictions

## File Structure
```
equipment-predictive-maintenance/
├── equip_predict/
│   ├── data_generator.py          # Synthetic sensor data
│   ├── models/
│   │   ├── failure_predictor.py   # 5-class status classification
│   │   ├── life_estimator.py      # RUL regression
│   │   └── anomaly_detector.py    # Isolation Forest
│   └── utils/
│       ├── preprocessor.py        # Feature engineering
│       ├── visualizer.py          # Charts
│       └── metrics.py             # Accuracy, F1, MAE, R2
├── scripts/
│   ├── train.py                   # Training pipeline
│   ├── predict.py                 # Equipment predictions
│   └── evaluate.py                # Full evaluation
├── templates/index.html           # Dashboard
├── app.py                         # Flask server
├── test_api.py                    # API tests
└── outputs/
    ├── models/                    # Trained models
    └── plots/                     # Generated charts
```
