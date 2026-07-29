import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Equipment Predictive Maintenance", layout="wide")
st.title("Equipment Predictive Maintenance")
st.markdown("Predict equipment failures and remaining useful life.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'failure': joblib.load(d / 'failure_classifier.pkl'), 'rul': joblib.load(d / 'rul_estimator.pkl')}

st.sidebar.header("Input Parameters")
vibration_x = st.sidebar.slider('Vibration X', 0, 50, 25)
vibration_y = st.sidebar.slider('Vibration Y', 0, 50, 25)
vibration_z = st.sidebar.slider('Vibration Z', 0, 50, 25)
temperature = st.sidebar.slider('Temperature', 20, 200, 110)
pressure = st.sidebar.slider('Pressure', 1, 100, 50)
flow = st.sidebar.slider('Flow', 0, 500, 250)
current = st.sidebar.slider('Current', 0, 200, 100)
rpm = st.sidebar.slider('Rpm', 0, 5000, 2500)
bearing_temp = st.sidebar.slider('Bearing Temp', 20, 150, 85)
oil_level = st.sidebar.slider('Oil Level', 0, 100, 50)
seal_pressure = st.sidebar.slider('Seal Pressure', 0, 50, 25)
suction_pressure = st.sidebar.slider('Suction Pressure', 0, 50, 25)
discharge_pressure = st.sidebar.slider('Discharge Pressure', 0, 50, 25)
power = st.sidebar.slider('Power', 0, 1000, 500)
operating_hrs = st.sidebar.slider('Operating Hrs', 0, 50000, 25000)
days_since_maint = st.sidebar.slider('Days Since Maint', 0, 365, 182)

if st.sidebar.button("Run"):
    try:
        x = np.array([[vibration_x, vibration_y, vibration_z, temperature, pressure, flow, current, rpm, bearing_temp, oil_level, seal_pressure, suction_pressure, discharge_pressure, power, operating_hrs, days_since_maint]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))