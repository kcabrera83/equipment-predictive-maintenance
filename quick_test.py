import matplotlib
matplotlib.use("Agg")
import time, sys
sys.path.insert(0, ".")

print("1. Generating data...", flush=True)
t = time.time()
from equip_predict.data_generator import EquipDataGenerator
gen = EquipDataGenerator(seed=42)
df = gen.generate(n_units=80, n_days=200)
print(f"   {len(df)} records in {time.time()-t:.1f}s", flush=True)

print("2. Preprocessing...", flush=True)
t = time.time()
from equip_predict.utils.preprocessor import EquipPreprocessor
prep = EquipPreprocessor()
X_cls_train, X_cls_test, y_cls_train, y_cls_test, cls_features = prep.prepare_classification(df)
X_rul_train, X_rul_test, y_rul_train, y_rul_test, rul_features = prep.prepare_rul(df)
print(f"   Done in {time.time()-t:.1f}s", flush=True)

print("3. Training classifiers...", flush=True)
from equip_predict.models.failure_predictor import FailurePredictor
for name in ["random_forest", "gradient_boosting", "extra_trees"]:
    t = time.time()
    p = FailurePredictor(model_name=name)
    p.train(X_cls_train, y_cls_train)
    y_pred = p.predict(X_cls_test)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_cls_test, y_pred)
    f1 = f1_score(y_cls_test, y_pred, average="macro", zero_division=0)
    print(f"   {name}: Acc={acc:.4f} F1={f1:.4f} ({time.time()-t:.1f}s)", flush=True)

print("4. Training RUL estimators...", flush=True)
from equip_predict.models.life_estimator import LifeEstimator
for name in ["random_forest", "gradient_boosting", "extra_trees"]:
    t = time.time()
    p = LifeEstimator(model_name=name)
    p.train(X_rul_train, y_rul_train)
    y_pred = p.predict(X_rul_test)
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_rul_test, y_pred)
    mae = mean_absolute_error(y_rul_test, y_pred)
    print(f"   {name}: R2={r2:.4f} MAE={mae:.2f} ({time.time()-t:.1f}s)", flush=True)

print("5. Anomaly detection...", flush=True)
t = time.time()
from equip_predict.models.anomaly_detector import EquipmentAnomalyDetector
detector = EquipmentAnomalyDetector()
df_anomaly, n_anomalies = detector.detect_anomalies(df)
summary = detector.get_anomaly_summary(df_anomaly)
print(f"   Anomalies: {summary['anomaly_count']} ({time.time()-t:.1f}s)", flush=True)

print("ALL DONE", flush=True)
