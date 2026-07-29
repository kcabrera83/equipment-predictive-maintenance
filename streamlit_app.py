
import streamlit as st
import numpy as np
import joblib, os
from typing import Callable, Dict

st.set_page_config(page_title="Equipment Predictive Maintenance", layout="centered")
st.title(":gear: Equipment Predictive Maintenance")
st.caption("Rotating equipment remaining useful life and failure prediction")

def pipe_load(path: str) -> dict:
    return joblib.load(path)

def pipe_scale(data: dict, features: list, inputs: list) -> np.ndarray:
    X = np.array(inputs).reshape(1, -1)
    scaler = data.get("scaler")
    return scaler.transform(X) if scaler else X

def pipe_predict(data: dict, X: np.ndarray) -> float:
    return data["model"].predict(X)[0]

MODEL_CACHE: Dict[str, dict] = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        MODEL_CACHE[f.replace(".pkl", "")] = pipe_load(os.path.join("outputs/models", f))

sel = st.selectbox("Select model pipeline", list(MODEL_CACHE.keys()) or ["default"])
data = MODEL_CACHE.get(sel, {})
feats = data.get("feature_names", [f"var_{i}" for i in range(4)])

with st.form("pipeline"):
    vals = [st.number_input(f, value=0.0) for f in feats]
    if st.form_submit_button("Execute pipeline"):
        Xs = pipe_scale(data, feats, vals)
        p = pipe_predict(data, Xs)
        st.balloons()
        st.success(f"Pipeline result: {p:.4f}")
