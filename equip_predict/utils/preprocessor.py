"""Preprocesamiento de datos de sensores para mantenimiento predictivo."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SENSOR_FEATURES = [
    "vibration_x", "vibration_y", "vibration_z",
    "temperature", "pressure", "flow_rate", "motor_current",
    "rpm", "bearing_temp", "oil_level", "seal_pressure",
    "suction_pressure", "discharge_pressure", "power_consumption",
    "runtime_hours", "days_since_maintenance",
]

DERIVED_FEATURES = [
    "vibration_total", "vibration_trend",
    "temp_pressure_ratio", "efficiency",
    "power_per_flow", "seal_pressure_ratio",
    "discharge_suction_diff", "bearing_temp_diff",
]


class EquipPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._fitted = False

    def _build_derived(self, df):
        df = df.copy()
        eps = 1e-6
        df["vibration_total"] = np.sqrt(df["vibration_x"]**2 + df["vibration_y"]**2 + df["vibration_z"]**2)
        df["vibration_trend"] = 0 if "unit_id" not in df.columns or len(df) < 2 else df.groupby("unit_id")["vibration_x"].transform(lambda x: x.diff(5).fillna(0))
        df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"] + eps)
        df["efficiency"] = df["flow_rate"] / (df["power_consumption"] + eps)
        df["power_per_flow"] = df["power_consumption"] / (df["flow_rate"] + eps)
        df["seal_pressure_ratio"] = df["seal_pressure"] / (df["pressure"] + eps)
        df["discharge_suction_diff"] = df["discharge_pressure"] - df["suction_pressure"]
        df["bearing_temp_diff"] = df["bearing_temp"] - df["temperature"]
        return df

    def get_feature_names(self):
        return SENSOR_FEATURES + DERIVED_FEATURES

    def fit(self, df):
        df = self._build_derived(df)
        features = self.get_feature_names()
        self.scaler.fit(df[features].fillna(0))
        self._fitted = True
        return self

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        df = self._build_derived(df)
        features = self.get_feature_names()
        X = df[features].fillna(0)
        return pd.DataFrame(self.scaler.transform(X), columns=features, index=df.index)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def prepare_classification(self, df, target_col="status"):
        df = self._build_derived(df)
        features = self.get_feature_names()
        X = df[features].fillna(0)
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y,
        )
        return X_train, X_test, y_train, y_test, list(X.columns)

    def prepare_rul(self, df):
        df = self._build_derived(df)
        features = self.get_feature_names()
        X = df[features].fillna(0)
        y = df["rul_days"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test, list(X.columns)
