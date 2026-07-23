"""Metricas de evaluacion para mantenimiento predictivo."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
)


class EvalMetrics:
    def __init__(self):
        self.results = {}

    def classification_metrics(self, y_true, y_pred, model_name="model"):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        self.results[model_name] = metrics
        return metrics

    def regression_metrics(self, y_true, y_pred, model_name="model"):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        try:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        except Exception:
            mape = 0
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}
        self.results[model_name] = metrics
        return metrics

    def print_classification_report(self, y_true, y_pred, class_names=None):
        import numpy as np
        labels = sorted(set(y_true) | set(y_pred))
        names = [class_names[i] for i in labels] if class_names else None
        print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))

    def print_confusion_matrix(self, y_true, y_pred, class_names=None):
        import numpy as np
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion Matrix:")
        names = [class_names[i][:8] if class_names and i < len(class_names) else str(i) for i in labels]
        print("  " + "".join(f"{n:>10}" for n in names))
        for i, row in enumerate(cm):
            print(f"  {names[i]:>8} " + "".join(f"{v:>10}" for v in row))

    def print_report(self):
        print(f"\n{'='*65}")
        print(f"  {'Modelo':<25} {'Metrica':>10} {'Valor':>10}")
        print(f"{'='*65}")
        for name, m in self.results.items():
            for metric, value in m.items():
                print(f"  {name:<25} {metric:>10} {value:>10.4f}")
        print(f"{'='*65}")
