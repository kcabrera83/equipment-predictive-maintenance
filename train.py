"""Training script for equipment predictive maintenance models."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equip_predict.data_generator import EquipDataGenerator
from equip_predict.models.failure_predictor import FailurePredictor
from equip_predict.models.life_estimator import LifeEstimator
from equip_predict.utils.preprocessor import EquipPreprocessor

ANOMALY_FEATURES = [
    "vibration_x", "vibration_y", "vibration_z",
    "temperature", "pressure", "motor_current",
    "bearing_temp", "power_consumption",
]


def main():
    print("=" * 60)
    print("  Equipment Predictive Maintenance - Training")
    print("=" * 60)

    print("\n[1/6] Generating synthetic sensor data...")
    gen = EquipDataGenerator(seed=42)
    df = gen.generate(n_units=150, n_days=365)
    os.makedirs("outputs/data", exist_ok=True)
    df.to_csv("outputs/data/equipment_sensors.csv", index=False)
    print(f"  Dataset: {len(df)} records, {df['unit_id'].nunique()} equipment units")

    print("\n[2/6] Preprocessing...")
    preprocessor = EquipPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_classification(df, target_col="status")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[3/6] Training failure predictor (classifier)...")
    predictor = FailurePredictor(model_name="gradient_boosting")
    predictor.train(X_train, y_train)
    train_pred = predictor.predict(X_train)
    test_pred = predictor.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test  Accuracy: {test_acc:.4f}")

    print("\n[4/6] Training RUL estimator (regressor)...")
    X_rul_train, X_rul_test, y_rul_train, y_rul_test, _ = preprocessor.prepare_rul(df)
    estimator = LifeEstimator(model_name="gradient_boosting")
    estimator.train(X_rul_train, y_rul_train)
    rul_train_pred = estimator.predict(X_rul_train)
    rul_test_pred = estimator.predict(X_rul_test)
    train_r2 = r2_score(y_rul_train, rul_train_pred)
    test_r2 = r2_score(y_rul_test, rul_test_pred)
    train_mae = mean_absolute_error(y_rul_train, rul_train_pred)
    test_mae = mean_absolute_error(y_rul_test, rul_test_pred)
    print(f"  Train R2: {train_r2:.4f} | MAE: {train_mae:.4f}")
    print(f"  Test  R2: {test_r2:.4f} | MAE: {test_mae:.4f}")

    print("\n[5/6] Training anomaly detector (Isolation Forest on normal data)...")
    normal_df = df[df["status"] == 0]
    X_anomaly = normal_df[ANOMALY_FEATURES].values
    anomaly_detector = IForest(contamination=0.05, random_state=42)
    anomaly_detector.fit(X_anomaly)
    print(f"  Fitted on {len(X_anomaly)} normal records")

    print("\n[6/6] Saving models and preprocessor...")
    os.makedirs("outputs/models", exist_ok=True)
    predictor.save("outputs/models/failure_predictor.pkl")
    estimator.save("outputs/models/life_estimator.pkl")
    with open("outputs/models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("outputs/models/anomaly_detector.pkl", "wb") as f:
        pickle.dump(anomaly_detector, f)

    print("  Models saved to outputs/models/")
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Failure Predictor:  Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")
    print(f"  RUL Estimator:      Train R2={train_r2:.4f} | Test R2={test_r2:.4f}")
    print(f"  Anomaly Detector:   Fitted on {len(X_anomaly)} normal samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
