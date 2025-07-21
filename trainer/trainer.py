#!/usr/bin/env python
"""Train a simple Titanic‑survival logistic‑regression model."""

from datetime import datetime
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -------- 参数 --------
parser = argparse.ArgumentParser()
parser.add_argument("--out", default="model/model.pkl")
parser.add_argument("--upload-minio", action="store_true")
args = parser.parse_args()

# -------- 训练 --------
url = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)
df = (
    pd.read_csv(url)[["Survived", "Sex", "Age", "Fare", "Pclass"]]
    .dropna()
)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
X, y = df[["Sex", "Age", "Fare", "Pclass"]], df["Survived"]

model = LogisticRegression()
model.fit(X, y)

# -------- 保存模型--------
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out_path)

meta = {
    "model_file": out_path.name,
    "created_at_utc": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    "n_samples": len(df),
    "n_features": X.shape[1],
    "features": list(X.columns),
}
meta_path = out_path.with_suffix(".json")
with meta_path.open("w") as f:
    json.dump(meta, f, indent=2)

print(f"model -> {out_path}")
print(f"meta  -> {meta_path}")

# -------- 可选：上传到 MinIO --------
if args.upload_minio:
    from minio import Minio  # noqa: WPS433 (allowed: import inside guard)

    client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    bucket = os.getenv("MINIO_BUCKET", "model")
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    for fp in (out_path, meta_path):
        client.fput_object(bucket, fp.name, str(fp))
        print(f"uploaded {fp.name} -> bucket {bucket}")
