import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from minio import Minio

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df[["Survived", "Sex", "Age", "Fare", "Pclass"]]
df = df.dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Sex", "Age", "Fare", "Pclass"]]
y = df["Survived"]

model = LogisticRegression()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

minio_client = Minio(
    "minio-service:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "model"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

minio_client.fput_object(
    bucket_name,
    "model.pkl",
    "model/model.pkl"
)
print("upload MinIO！")
