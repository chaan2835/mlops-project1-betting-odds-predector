#!/bin/bash

mkdir -p /tmp/mlflow

#################################################
# Train model if not available
#################################################
if [ ! -f artifacts/models/model.pkl ]; then
    echo "Model not found. Training model..."
    python main.py
else
    echo "Model already exists. Skipping training."
fi

echo "Starting MLflow..."

mlflow server \
--host 0.0.0.0 \
--port 5001 \
--backend-store-uri sqlite:////tmp/mlflow/mlflow.db \
--default-artifact-root /tmp/mlflow/artifacts \
--allowed-hosts "*" \
--cors-allowed-origins "*" &

echo "Starting FastAPI..."

uvicorn api.app:app \
--host 0.0.0.0 \
--port 8000 &

echo "Starting Streamlit..."

streamlit run streamlit_app/app.py \
--server.port 8502 \
--server.address 0.0.0.0