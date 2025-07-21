import os
import json
import requests
import pandas as pd
import streamlit as st
from minio import Minio

SERVING_URL = os.getenv("SERVING_URL", "http://serving-service:8000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "model"

mode = st.sidebar.radio("Mode", ["Register", "Login"])

if "auth" not in st.session_state:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Submit"):
        url = f"{SERVING_URL}/{mode.lower()}"
        r = requests.post(url, json={"username": username, "password": password})
        if r.status_code == 200:
            if mode == "Register":
                st.success("Registration successful. Please switch to Login.")
            else:
                st.session_state.auth = True
        else:
            st.error(r.text)
    st.stop()

st.set_page_config(page_title="Titanic Predictor", layout="wide")
st.title("Titanic Survival Predictor")

try:
    metrics = requests.get(f"{SERVING_URL}/metrics/summary", timeout=3).json()
    c1, c2 = st.columns(2)
    c1.metric("p50 Latency", f"{metrics['lat50']} ms")
    c2.metric("Requests Today", metrics["req_today"])
    st.caption(f"Model version: {metrics['model_version']}")
except:
    st.info("Metrics unavailable")

with st.sidebar:
    st.header("Passenger Info")
    sex = st.selectbox(
        "Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female"
    )
    age = st.number_input("Age", 0.0, 100.0, 30.0)
    fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    pclass = st.selectbox("Pclass", [1, 2, 3])
    if st.button("Predict"):
        payload = {"Sex": sex, "Age": age, "Fare": fare, "Pclass": pclass}
        try:
            r = requests.post(f"{SERVING_URL}/predict", json=payload, timeout=5)
            r.raise_for_status()
            pred = r.json()["predict"]
            st.session_state.setdefault("hist", []).append(pred)
            st.success("Survived" if pred == 1 else "Did NOT Survive")
        except Exception as e:
            st.error(f"Serving error: {e}")

st.markdown("---")
st.subheader("Cumulative Survival Rate")
hist = st.session_state.get("hist", [])
if hist:
    df_hist = pd.DataFrame(
        {
            "attempt": range(1, len(hist) + 1),
            "survival_rate": pd.Series(hist).expanding().mean() * 100,
        }
    )
    st.line_chart(df_hist.set_index("attempt"), height=250, use_container_width=True)
else:
    st.info("Run predictions to see the chart")

st.markdown("---")
st.subheader("Model Registry")
try:
    mc = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=False)
    metas = []
    for obj in mc.list_objects(BUCKET, recursive=True):
        if obj.object_name.endswith(".json"):
            meta = json.loads(mc.get_object(BUCKET, obj.object_name).read().decode())
            metas.append((obj.object_name, meta))
    if metas:
        for fname, meta in sorted(
            metas, key=lambda x: meta.get("created_at", ""), reverse=True
        ):
            with st.expander(fname, expanded=False):
                st.write("Created:", meta.get("created_at"))
                st.write("Samples:", meta.get("n_samples"))
                st.json(meta)
    else:
        st.info("No model metadata found")
except Exception as e:
    st.error(f"MinIO error: {e}")
