#!/bin/bash

echo "Starting MLflow..."

mlflow server \
--host 0.0.0.0 \
--port 5001 \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns &

echo "Starting FastAPI..."

uvicorn api.app:app \
--host 0.0.0.0 \
--port 8000 &

echo "Starting Streamlit..."

streamlit run streamlit_app/app.py \
--server.port 8502 \
--server.address 0.0.0.0