#!/usr/bin/env python3
"""Script de entrenamiento de modelos de mantenimiento predictivo."""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from pyod.models.iforest import IForest
from lifelines import WeibullFitter

from equip_predict.data_generator import EquipDataGenerator, STATUS_LABELS
from equip_predict.utils.preprocessor import EquipPreprocessor
from equip_predict.utils.metrics import EvalMetrics
from equip_predict.utils.visualizer import EquipVisualizer
from equip_predict.models.failure_predictor import FailurePredictor
from equip_predict.models.life_estimator import LifeEstimator


def train_failure_predictor(X, y):
    """Train XGBoost classifier for failure prediction."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_anomaly_detector(X):
    """Train Isolation Forest from PyOD for anomaly detection."""
    model = IForest(contamination=0.05, random_state=42)
    model.fit(X)
    return model


def train_survival_model(durations, events):
    """Train Weibull AFT model for remaining useful life."""
    wf = WeibullFitter()
    durations = np.maximum(durations, 1.0)
    events = events.astype(bool)
    wf.fit(durations, event_observed=events)
    return wf


def main():
    print("=" * 60)
    print("  TRAINING - Equipment Predictive Maintenance")
    print("=" * 60)

    gen = EquipDataGenerator(seed=42)
    df = gen.generate(n_units=50, n_days=120)
    gen.save(df)
    print(f"  Dataset: {len(df)} records, {df['unit_id'].nunique()} units")

    prep = EquipPreprocessor()
    viz = EquipVisualizer()

    viz.plot_failure_distribution(df)
    viz.plot_rul_distribution(df)
    viz.plot_correlation(df)
    print("  Plots generated in outputs/plots/")

    print("\n  [1] Training failure classification (XGBoost)...")
    X_cls_train, X_cls_test, y_cls_train, y_cls_test, cls_features = prep.prepare_classification(df)
    print(f"  Features: {len(cls_features)} | Train: {len(X_cls_train)} | Test: {len(X_cls_test)}")

    all_cls_models = FailurePredictor.train_all(X_cls_train, y_cls_train)
    xgb_model = train_failure_predictor(X_cls_train, y_cls_train)
    all_cls_models["XGBoost"] = xgb_model

    cls_eval = EvalMetrics()
    best_cls_name, best_cls_model = None, None

    for name, model in all_cls_models.items():
        y_pred = model.predict(X_cls_test)
        metrics = cls_eval.classification_metrics(y_cls_test, y_pred, name)
        print(f"  {name:<25} Acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}")
        if best_cls_name is None or metrics["f1_macro"] > cls_eval.results[best_cls_name]["f1_macro"]:
            best_cls_name, best_cls_model = name, model

    print(f"\n  Best classifier: {best_cls_name}")
    class_names = [STATUS_LABELS[i] for i in sorted(STATUS_LABELS.keys())]
    y_best_pred = best_cls_model.predict(X_cls_test)
    cls_eval.print_classification_report(y_cls_test, y_best_pred, class_names)
    cls_eval.print_confusion_matrix(y_cls_test, y_best_pred, class_names)
    best_cls_model.save("outputs/models/failure_predictor.pkl")

    print("\n  [2] Training RUL estimation models...")
    X_rul_train, X_rul_test, y_rul_train, y_rul_test, rul_features = prep.prepare_rul(df)
    print(f"  Features: {len(rul_features)} | Train: {len(X_rul_train)} | Test: {len(X_rul_test)}")

    all_rul_models = LifeEstimator.train_all(X_rul_train, y_rul_train)
    rul_eval = EvalMetrics()
    best_rul_name, best_rul_model = None, None

    for name, model in all_rul_models.items():
        y_pred = model.predict(X_rul_test)
        metrics = rul_eval.regression_metrics(y_rul_test, y_pred, name)
        print(f"  {name:<25} R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.2f}")
        if best_rul_name is None or metrics["R2"] > rul_eval.results[best_rul_name]["R2"]:
            best_rul_name, best_rul_model = name, model

    print(f"\n  Best RUL estimator: {best_rul_name}")
    y_rul_pred = best_rul_model.predict(X_rul_test)
    viz.plot_predictions_vs_actual(y_rul_test, y_rul_pred, title=f"RUL Predictions - {best_rul_name}")
    rul_eval.print_report()
    best_rul_model.save("outputs/models/life_estimator.pkl")

    print("\n  [3] Training anomaly detector (PyOD IForest)...")
    X_anomaly = X_cls_train.copy()
    anomaly_detector = train_anomaly_detector(X_anomaly)
    anomaly_preds = anomaly_detector.predict(X_anomaly)
    n_anomalies = (anomaly_preds == 1).sum()
    print(f"  PyOD IForest detected {n_anomalies}/{len(X_anomaly)} anomalies in training set")

    print("\n  [4] Training survival model (WeibullFitter)...")
    if "rul_days" in df.columns:
        durations = df["rul_days"].values.astype(float)
        events = (df["failure"].values.astype(bool) if "failure" in df.columns
                  else np.ones(len(durations), dtype=bool))
        survival_model = train_survival_model(durations, events)
        print(f"  WeibullFitter trained: lambda={survival_model.rho_:.4f}, rho={survival_model.lambda_:.4f}")

    print("\n  Training completed successfully!")
    return best_cls_name, best_rul_name


if __name__ == "__main__":
    main()
