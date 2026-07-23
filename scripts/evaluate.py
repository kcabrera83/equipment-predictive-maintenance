#!/usr/bin/env python3
"""Script de evaluacion de mantenimiento predictivo."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equip_predict.data_generator import EquipDataGenerator, STATUS_LABELS
from equip_predict.utils.preprocessor import EquipPreprocessor
from equip_predict.utils.metrics import EvalMetrics
from equip_predict.utils.visualizer import EquipVisualizer
from equip_predict.models.failure_predictor import FailurePredictor
from equip_predict.models.life_estimator import LifeEstimator
from equip_predict.models.anomaly_detector import EquipmentAnomalyDetector


def main():
    print("=" * 60)
    print("  EVALUATION - Equipment Predictive Maintenance")
    print("=" * 60)

    gen = EquipDataGenerator(seed=42)
    df = gen.generate(n_units=50, n_days=120)
    prep = EquipPreprocessor()
    viz = EquipVisualizer()
    class_names = [STATUS_LABELS[i] for i in sorted(STATUS_LABELS.keys())]

    print("\n  [1] Classification evaluation with cross-validation...")
    X_cls_train, X_cls_test, y_cls_train, y_cls_test, _ = prep.prepare_classification(df)
    all_cls = FailurePredictor.train_all(X_cls_train, y_cls_train)
    cls_eval = EvalMetrics()
    for name, model in all_cls.items():
        cv = model.cross_validate(X_cls_train, y_cls_train, cv=3)
        y_pred = model.predict(X_cls_test)
        metrics = cls_eval.classification_metrics(y_cls_test, y_pred, name)
        print(f"  {name:<25} CV-F1={cv['mean_f1']:.4f}  Test-Acc={metrics['accuracy']:.4f}")
    print("\n  Classification Report (best model):")
    best_cls = max(cls_eval.results, key=lambda k: cls_eval.results[k]["f1_macro"])
    y_best = all_cls[best_cls].predict(X_cls_test)
    cls_eval.print_classification_report(y_cls_test, y_best, class_names)
    cls_eval.print_confusion_matrix(y_cls_test, y_best, class_names)

    print("\n  [2] RUL estimation evaluation...")
    X_rul_train, X_rul_test, y_rul_train, y_rul_test, _ = prep.prepare_rul(df)
    all_rul = LifeEstimator.train_all(X_rul_train, y_rul_train)
    rul_eval = EvalMetrics()
    for name, model in all_rul.items():
        y_pred = model.predict(X_rul_test)
        metrics = rul_eval.regression_metrics(y_rul_test, y_pred, name)
        print(f"  {name:<25} R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.2f}")
    rul_eval.print_report()
    best_rul = rul_eval.best_model() if hasattr(rul_eval, 'best_model') else max(rul_eval.results, key=lambda k: rul_eval.results[k]["R2"])
    y_rul_pred = all_rul[best_rul].predict(X_rul_test)
    viz.plot_predictions_vs_actual(y_rul_test, y_rul_pred, title=f"RUL Evaluation - {best_rul}")

    print("\n  [3] Anomaly detection...")
    detector = EquipmentAnomalyDetector()
    df_anomaly, n_anomalies = detector.detect_anomalies(df)
    summary = detector.get_anomaly_summary(df_anomaly)
    print(f"  Anomalies detected: {summary['anomaly_count']} ({summary['anomaly_pct']}%)")

    print("\n  [4] Generating plots...")
    viz.plot_failure_distribution(df)
    viz.plot_rul_distribution(df)
    viz.plot_model_comparison(cls_eval.results)
    viz.plot_sensor_trends(df)

    eval_report = {
        "best_classifier": best_cls,
        "classifier_metrics": cls_eval.results[best_cls],
        "best_rul_estimator": best_rul,
        "rul_metrics": rul_eval.results[best_rul],
        "anomaly_summary": summary,
    }
    output_path = Path("outputs/evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Report saved to {output_path}")
    print("  Evaluation completed successfully!")
    return cls_eval, rul_eval


if __name__ == "__main__":
    main()
