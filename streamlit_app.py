import streamlit as st
import joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance")
st.markdown("RUL estimation & failure classification")

@st.cache_resource
def load_models():
    base = Path(__file__).parent / 'outputs' / 'models'
    return {'state': joblib.load(base / 'failure_classifier.pkl'), 'rul': joblib.load(base / 'rul_estimator.pkl')}

models = load_models()

def predict(name, x):
    m = models[name]
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        if 'label_encoder' in m:
            return m['label_encoder'].inverse_transform(p)[0]
        return float(p[0])
    return float(m.predict(x)[0])

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader('Parameters')
    vib_x = st.slider('Vib X', 0, 50, 25)
    vib_y = st.slider('Vib Y', 0, 50, 25)
    vib_z = st.slider('Vib Z', 0, 50, 25)
    temp = st.slider('Temp', 20, 200, 110)
    pres = st.slider('Pres', 1, 100, 50)
    flow = st.slider('Flow', 0, 500, 250)
    curr = st.slider('Curr', 0, 200, 100)
    rpm = st.slider('Rpm', 0, 5000, 2500)
    bearing = st.slider('Bearing', 20, 150, 85)
    oil = st.slider('Oil', 0, 100, 50)
    run = st.button('Run Prediction', use_container_width=True)

with col2:
    if run:
        x = np.array([[vib_x, vib_y, vib_z, temp, pres, flow, curr, rpm, bearing, oil]])
        results = {}
        results['state'] = predict('state', x)
        results['rul'] = predict('rul', x)
        st.subheader('Results')
        rcols = st.columns(len(results))
        for i, (k, v) in enumerate(results.items()):
            label = k.replace('_', ' ').title()
            if isinstance(v, str):
                rcols[i].metric(label, v)
            else:
                rcols[i].metric(label, f'{v:.2f}')
        # Plot
        fig, ax = plt.subplots()
        names = [k.replace('_',' ').title() for k in results]
        vals = [float(v) if isinstance(v, (int,float,str)) and str(v).replace('.','').replace('-','').isdigit() else 0 for v in results.values()]
        if any(v != 0 for v in vals):
            ax.bar(names, vals, color=['#0077B6','#00B4D8','#90E0EF'])
            ax.set_ylabel('Value')
            st.pyplot(fig)