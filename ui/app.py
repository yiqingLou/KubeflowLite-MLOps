import os
import json
from typing import List

import requests
import streamlit as st
import pandas as pd
from minio import Minio

SERVING_URL = os.getenv("SERVING_URL", "http://serving-service:8000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "model"

st.set_page_config(page_title="Titanic Predictor", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        font-size: 16px !important;
        font-weight: bold !important;
        font-family: Arial, Helvetica, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

mode = st.sidebar.radio("Mode", ["Register", "Login"])


if "auth" not in st.session_state:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Submit"):
        try:
            url = f"{SERVING_URL}/{mode.lower()}"
            resp = requests.post(
                url, json={"username": username, "password": password}, timeout=5
            )
            if resp.status_code == 200:
                if mode == "Register":
                    st.success("Registration successful, switch to Login.")
                else:
                    st.session_state.auth = True
            else:
                st.error(resp.text)
        except requests.RequestException as err:
            st.error(f"Cannot reach backend: {err}")
    st.stop()


try:
    metrics = requests.get(f"{SERVING_URL}/metrics/summary", timeout=3).json()
    c1, c2 = st.columns(2)
    c1.metric("p50 Latency", f"{metrics['lat50']} ms")
    c2.metric("Requests Today", metrics["req_today"])
    st.caption(f"Model version: {metrics['model_version']}")
except requests.RequestException:
    st.info("Metrics unavailable")

with st.sidebar:
    st.header("Passenger Info")
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
            st.success("Survived" if pred == 1 else "Did NOT Survive")
        except requests.RequestException as err:
            st.error(f"Serving error: {err}")

st.markdown("---")
st.subheader("Cumulative Survival Rate")
history: List[int] = st.session_state.get("hist", [])
if history:
    df_hist = pd.DataFrame(
        {
            "attempt": range(1, len(history) + 1),
            "survival_rate": pd.Series(history).expanding().mean() * 100,
        }
    ).set_index("attempt")
    st.line_chart(df_hist, height=250, use_container_width=True)
else:
    st.info("Run predictions to see the chart")

st.markdown("---")
st.subheader("Model Registry")
try:
    mc = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=False)
    metas = []
    for obj in mc.list_objects(BUCKET, recursive=True):
        if obj.object_name.endswith(".json"):
            meta = json.loads(mc.get_object(BUCKET, obj.object_name).read().decode())
            metas.append((obj.object_name, meta))
    if metas:
        for fname, meta in sorted(
            metas, key=lambda x: x[1].get("created_at", ""), reverse=True
        ):
            with st.expander(fname, expanded=False):
                st.write("Created:", meta.get("created_at"))
                st.write("Samples:", meta.get("n_samples"))
                st.json(meta)
    else:
        st.info("No model metadata found")
except Exception as err:
    st.error(f"MinIO error: {err}")
