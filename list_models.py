from minio import Minio
import json

minio_client = Minio(
    "localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False
)

bucket_name = "model"
objects = minio_client.list_objects(bucket_name, recursive=True)

for obj in objects:
    if obj.object_name.endswith(".json"):
        data = minio_client.get_object(bucket_name, obj.object_name)
        metadata = json.loads(data.read().decode("utf-8"))
        print(f"{obj.object_name}:")
        print(json.dumps(metadata, indent=2))
        print("-" * 40)
