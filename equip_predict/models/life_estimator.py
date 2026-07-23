"""Modelo de estimacion de Vida Util Restante (RUL)."""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score


class LifeEstimator:
    MODELS = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.15, subsample=0.8, random_state=42,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=80, max_depth=10, random_state=42, n_jobs=-1,
        ),
    }

    def __init__(self, model_name="gradient_boosting"):
        if model_name not in self.MODELS:
            raise ValueError(f"Model '{model_name}' not available.")
        self.model_name = model_name
        self.model = self.MODELS[model_name]()
        self._trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Model not trained.")
        return np.maximum(self.model.predict(X), 0)

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return {"mean_mae": -scores.mean(), "std_mae": scores.std()}

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None

    def save(self, path="outputs/models/life_estimator.pkl"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "model_name": self.model_name}, filepath)
        return filepath

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        instance = cls(model_name=data["model_name"])
        instance.model = data["model"]
        instance._trained = True
        return instance

    @staticmethod
    def train_all(X_train, y_train):
        results = {}
        for name in LifeEstimator.MODELS:
            print(f"  Training {name}...")
            p = LifeEstimator(model_name=name)
            p.train(X_train, y_train)
            results[name] = p
        return results
