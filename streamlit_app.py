import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Equipment Predictive Maintenance", layout="wide")
st.title("Equipment Predictive Maintenance")
st.markdown("Predict equipment failures, estimate remaining useful life, and detect sensor anomalies.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("failure", "failure_classifier.pkl"), ("rul", "rul_estimator.pkl"), ("anomaly", "anomaly_detector.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
vibration_x_mm = st.sidebar.slider("Vibration X Mm", 0, 50, 25)
vibration_y_mm = st.sidebar.slider("Vibration Y Mm", 0, 50, 25)
vibration_z_mm = st.sidebar.slider("Vibration Z Mm", 0, 50, 25)
temperature_c = st.sidebar.slider("Temperature C", 20, 200, 110)
pressure_bar = st.sidebar.slider("Pressure Bar", 1, 100, 50)
flow_rate_m3h = st.sidebar.slider("Flow Rate M3H", 0, 500, 250)
current_a = st.sidebar.slider("Current A", 0, 200, 100)
rpm = st.sidebar.slider("Rpm", 0, 5000, 2500)
bearing_temp_c = st.sidebar.slider("Bearing Temp C", 20, 150, 85)
oil_level_pct = st.sidebar.slider("Oil Level Pct", 0, 100, 50)
seal_pressure_bar = st.sidebar.slider("Seal Pressure Bar", 0, 50, 25)
suction_pressure_bar = st.sidebar.slider("Suction Pressure Bar", 0, 50, 25)
discharge_pressure_bar = st.sidebar.slider("Discharge Pressure Bar", 0, 50, 25)
power_kw = st.sidebar.slider("Power Kw", 0, 1000, 500)
operating_hrs = st.sidebar.slider("Operating Hrs", 0, 50000, 25000)
days_since_maint = st.sidebar.slider("Days Since Maint", 0, 365, 182)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[vibration_x_mm, vibration_y_mm, vibration_z_mm, temperature_c, pressure_bar, flow_rate_m3h, current_a, rpm, bearing_temp_c, oil_level_pct, seal_pressure_bar, suction_pressure_bar, discharge_pressure_bar, power_kw, operating_hrs, days_since_maint]])
        m = models["failure"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Failure", result if isinstance(result, str) else f"{result:.4f}")
        m = models["rul"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Rul", result if isinstance(result, str) else f"{result:.4f}")
        m = models["anomaly"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Anomaly", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

