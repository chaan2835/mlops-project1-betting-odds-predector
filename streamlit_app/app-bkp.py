import json
import os
import subprocess
import webbrowser

from dotenv import load_dotenv
from PIL import Image

import streamlit as st

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")
MLFLOW_URL = os.getenv("MLFLOW_URL")

if not FASTAPI_URL:
    raise ValueError("FASTAPI_URL is not set in the .env file.")

if not MLFLOW_URL:
    raise ValueError("MLFLOW_URL is not set in the .env file.")

METRICS_FILE = "artifacts/reports/metrics.json"

CONFUSION_MATRIX = "artifacts/plots/confusion_matrix.png"

ROC_CURVE = "artifacts/plots/roc_curve.png"

FEATURE_IMPORTANCE = "artifacts/reports/feature_importance.csv"

st.set_page_config(
    page_title="Sports Betting MLOps Dashboard",
    layout="wide"
)

st.title("⚽ Sports Betting MLOps Dashboard")

st.divider()

###########################################################
# TOP BUTTONS
###########################################################

col1, col2, col3 = st.columns(3)

###########################################################
# FASTAPI
###########################################################

with col1:

    st.subheader("FastAPI")

    if st.button("Start FastAPI"):

        subprocess.Popen([
            "uvicorn",
            "api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000"
        ])

        st.success("FastAPI Started")

    st.link_button(
        "Open Swagger",
        FASTAPI_URL + "/docs"
    )

###########################################################
# MLFLOW
###########################################################

with col2:

    st.subheader("MLflow")

    if st.button("Start MLflow"):

        subprocess.Popen([

            "mlflow",

            "server",

            "--host",

            "0.0.0.0",

            "--port",

            "5001"

        ])

        st.success("MLflow Started")

    st.link_button(
        "Open MLflow",
        MLFLOW_URL
    )

###########################################################
# ARTIFACTS
###########################################################

with col3:

    st.subheader("Artifacts")

    if st.button("Open Artifacts"):

        webbrowser.open("artifacts")

st.divider()

###########################################################
# MODEL METRICS
###########################################################

st.header("Model Metrics")

if os.path.exists(METRICS_FILE):

    with open(METRICS_FILE) as file:

        metrics = json.load(file)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        metrics["accuracy"]
    )

    c2.metric(
        "Precision",
        metrics["precision"]
    )

    c3.metric(
        "Recall",
        metrics["recall"]
    )

    c4.metric(
        "F1 Score",
        metrics["f1_score"]
    )

    c5.metric(
        "ROC AUC",
        metrics["roc_auc"]
    )

st.divider()

###########################################################
# REPORTS
###########################################################

st.header("Evaluation Reports")

left, right = st.columns(2)

with left:

    if os.path.exists(CONFUSION_MATRIX):

        st.image(
            Image.open(CONFUSION_MATRIX),
            caption="Confusion Matrix"
        )

with right:

    if os.path.exists(ROC_CURVE):

        st.image(
            Image.open(ROC_CURVE),
            caption="ROC Curve"
        )

st.divider()

###########################################################
# FEATURE IMPORTANCE
###########################################################

st.header("Feature Importance")

if os.path.exists(FEATURE_IMPORTANCE):

    st.dataframe(
        __import__("pandas").read_csv(
            FEATURE_IMPORTANCE
        ),
        use_container_width=True
    )