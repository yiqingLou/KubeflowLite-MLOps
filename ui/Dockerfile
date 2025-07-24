# ui/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先只复制 requirements，利用缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制剩余所有源码（包括 app.py）
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
