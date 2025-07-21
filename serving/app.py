import io
import json
import os
import time
from collections import deque
from datetime import date
from hashlib import sha256
from typing import List

import joblib
from fastapi import Body, FastAPI, HTTPException
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("MODEL_BUCKET", "model")
MODEL_OBJECT = os.getenv("MODEL_OBJECT")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def _choose_latest() -> str:
    objs = (
        o
        for o in client.list_objects(BUCKET_NAME, recursive=True)
        if o.object_name.endswith(".pkl")
    )
    return max(objs, key=lambda o: o.last_modified).object_name


def _download(obj: str) -> str:
    os.makedirs("model", exist_ok=True)
    path = os.path.join("model", os.path.basename(obj))
    client.fget_object(BUCKET_NAME, obj, path)
    return path


try:
    obj_name = MODEL_OBJECT or "model.pkl"
    try:
        model_path = _download(obj_name)
    except S3Error:
        obj_name = _choose_latest()
        model_path = _download(obj_name)
    model = joblib.load(model_path)
    load_error = None
except Exception as e:
    model, load_error = None, e

lat_hist = deque(maxlen=1000)
req_counter = {"date": date.today(), "count": 0}

app = FastAPI()


class Input(BaseModel):
    Sex: int = Field(..., ge=0, le=1)
    Age: float = Field(..., ge=0, le=120)
    Fare: float = Field(..., ge=0)
    Pclass: int = Field(..., ge=1, le=3)


class PredictOut(BaseModel):
    predict: int


class RegisterIn(BaseModel):
    username: str
    password: str


USERS_KEY = "users.json"


def _load_users() -> dict:
    try:
        data = client.get_object(BUCKET_NAME, USERS_KEY)
        return json.loads(data.read().decode())
    except Exception:
        return {}


def _save_users(users: dict):
    b = json.dumps(users).encode()
    client.put_object(
        BUCKET_NAME,
        USERS_KEY,
        io.BytesIO(b),
        length=len(b),
        content_type="application/json",
    )


@app.post("/register")
def register(r: RegisterIn = Body(...)):
    users = _load_users()
    if r.username in users:
        raise HTTPException(400, "User already exists")
    users[r.username] = sha256(r.password.encode()).hexdigest()
    _save_users(users)
    return {"status": "ok"}


@app.post("/login")
def login(r: RegisterIn = Body(...)):
    users = _load_users()
    h = sha256(r.password.encode()).hexdigest()
    if users.get(r.username) != h:
        raise HTTPException(401, "Bad credentials")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(inp: Input):
    if model is None:
        raise HTTPException(500, str(load_error))
    start = time.perf_counter()
    X = [[inp.Sex, inp.Age, inp.Fare, inp.Pclass]]
    y_pred: List[int] = model.predict(X)
    lat_hist.append((time.perf_counter() - start) * 1000)
    today = date.today()
    if req_counter["date"] != today:
        req_counter.update({"date": today, "count": 1})
    else:
        req_counter["count"] += 1
    return {"predict": int(y_pred[0])}


@app.get("/metrics/summary")
def metrics_summary():
    p50 = round(sorted(lat_hist)[len(lat_hist) // 2], 1) if lat_hist else 0
    return {
        "lat50": p50,
        "req_today": req_counter["count"],
        "model_version": obj_name,
    }


@app.post("/reload")
def reload_model():
    global model, load_error, obj_name
    try:
        obj_name = MODEL_OBJECT or _choose_latest()
        model = joblib.load(_download(obj_name))
        load_error = None
        return {"status": "reloaded", "object": obj_name}
    except Exception as e:
        load_error = e
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
