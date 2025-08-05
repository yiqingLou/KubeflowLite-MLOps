# KubeflowLite – Kubernetes-Native MLOps System

## 🔗 Demo Access

- 🌐 **Streamlit UI (Web Dashboard)**:  
  [http://104.197.168.94:8501](http://104.197.168.94:8501)  
  or  
  [http://35.232.77.229:8501](http://35.232.77.229:8501)

  > Use this dashboard to trigger training jobs, view model versions, test predictions, and promote models to production.

---

## 💡 Project Overview

KubeflowLite is a lightweight, open-source MLOps system that simulates a real-world machine learning workflow — from training to deployment to monitoring — on top of Kubernetes.

Compared to fully managed platforms like Vertex AI or the full Kubeflow stack, KubeflowLite is easier to set up, more resource-efficient, and ideal for students or early-career engineers to showcase their DevOps, system design, and ML engineering skills.

---

## 🔧 System Components

### 🏋️ Training Job
- A simple script `trainer.py` trains a logistic regression model using the Titanic dataset.
- Trained model (`model.pkl`) is uploaded to MinIO (S3-compatible storage).
- Each training run is isolated using a Kubernetes Job, simulating real-world job submission and reproducibility.

### 🗃️ Model Registry
- Stores all trained models in MinIO with versioning.
- Metadata such as timestamp, version, and accuracy are tracked.
- Enables rollback, auditing, and deployment of specific model versions.

### 🚀 Model Serving
- A FastAPI app exposes a `/predict` endpoint.
- Can serve either the latest model or a specified version.
- Deployed as a Kubernetes Deployment + Service.

### 📈 Monitoring Stack
- Prometheus scrapes metrics from trainer and serving components.
- Grafana visualizes key metrics like latency and accuracy.
- Loki collects logs from Kubernetes pods.

### 🖥️ UI Dashboard
- A Streamlit app provides a friendly interface for:
  - Launching new training jobs
  - Browsing model versions
  - Testing prediction outputs
  - Triggering deployment of specific models

---

## ⚙️ CI/CD & GitOps

- **GitHub Actions**: Automatically builds and pushes Docker images.
- **ArgoCD**: Automatically deploys images to Kubernetes clusters using Kustomize or Helm.

---

## 🧪 How to Run Locally

> This project is Kubernetes-native and consists of several components. Below are the key steps to run or interact locally:

### 1. Port Forward Services

```bash
# Serve Streamlit UI locally on port 8501
cd ui
streamlit run app.py

# Forward MinIO service
kubectl port-forward svc/minio 9000:9000

# Forward model serving API
kubectl port-forward svc/serving-service 8000:8000
