"""Detector de anomalias en sensores de equipos."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class EquipmentAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.scaler = StandardScaler()
        self._fitted = False
        self.feature_names = None

    def fit(self, df, features=None):
        if features is None:
            features = [
                "vibration_x", "vibration_y", "vibration_z",
                "temperature", "pressure", "motor_current",
                "bearing_temp", "power_consumption",
            ]
        self.feature_names = features
        X = df[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, df):
        if not self._fitted:
            raise RuntimeError("Detector must be fitted first.")
        X = df[self.feature_names].fillna(0).values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        return predictions, scores

    def detect_anomalies(self, df):
        if not self._fitted:
            self.fit(df)
        predictions, scores = self.predict(df)
        df = df.copy()
        df["anomaly"] = predictions
        df["anomaly_score"] = scores
        df["is_anomaly"] = predictions == -1
        n_anomalies = (predictions == -1).sum()
        return df, n_anomalies

    def get_anomaly_summary(self, df):
        anomalies = df[df["is_anomaly"]]
        return {
            "total_records": len(df),
            "anomaly_count": len(anomalies),
            "anomaly_pct": round(len(anomalies) / len(df) * 100, 2),
            "avg_anomaly_score": round(float(anomalies["anomaly_score"].mean()), 4) if len(anomalies) > 0 else 0,
        }
