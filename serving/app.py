import os
from minio import Minio
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

minio_client = Minio(
    "minio-service:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "model"
object_name = "model.pkl"
local_path = "model/model.pkl"
os.makedirs("model", exist_ok=True)
minio_client.fget_object(bucket_name, object_name, local_path)
model = joblib.load(local_path)

app = FastAPI()

class Input(BaseModel):
    Sex: int
    Age: float
    Fare: float
    Pclass: int

@app.post("/predict")
def predict(input: Input):
    X = [[input.Sex, input.Age, input.Fare, input.Pclass]]
    y_pred = model.predict(X)
    return {"predict": int(y_pred[0])}
