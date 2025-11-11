
import streamlit as st
import pandas as pd
from joblib import load
import json

st.set_page_config(page_title="Student Score Predictor", layout="centered")

st.title("🎯 Student Score Predictor")
st.write("Drop in your trained scikit-learn Pipeline (`artifacts/model.joblib`) and `artifacts/columns.json`.")

@st.cache_resource
def load_artifacts():
    try:
        pipe = load("artifacts/model.joblib")
        with open("artifacts/columns.json") as f:
            cols = json.load(f)
        return pipe, cols
    except Exception as e:
        st.warning(f"Artifacts not found or invalid: {e}")
        return None, None

pipe, cols = load_artifacts()

if pipe is None or cols is None:
    st.stop()

with st.form("predict_form"):
    st.subheader("Enter Features")
    # Create basic UI from columns (string inputs by default; pipeline should handle encoders)
    inputs = {}
    for c in cols:
        # Heuristic: numeric-like columns get number input, else text
        if any(x in c.lower() for x in ["hours", "attendance", "scores", "sessions", "income_num", "activity"]):
            inputs[c] = st.number_input(c, value=0.0, step=1.0)
        else:
            inputs[c] = st.text_input(c, value="")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        df = pd.DataFrame([inputs])[cols]  # enforce order
        pred = pipe.predict(df)[0]
        st.success(f"Predicted Exam Score: **{pred:.2f}**")
    except Exception as e:
        st.error(f"Inference error: {e}")
