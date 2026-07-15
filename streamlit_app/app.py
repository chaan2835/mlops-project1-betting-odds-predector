import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

###############################################################
# PAGE CONFIG
###############################################################

st.set_page_config(
    page_title="Sports Betting MLOps Dashboard",
    page_icon="⚽",
    layout="wide"
)

###############################################################
# ENV VARIABLES
###############################################################

FASTAPI_URL = os.getenv("FASTAPI_URL")
MLFLOW_URL = os.getenv("MLFLOW_URL")

###############################################################
# PROJECT PATHS
###############################################################

METRICS_PATH = "artifacts/reports/metrics.json"

MODEL_PATH = "artifacts/models/model.pkl"

PREPROCESSOR_PATH = "artifacts/models/preprocessor.pkl"

ROC_CURVE = "artifacts/plots/roc_curve.png"

CONFUSION_MATRIX = "artifacts/plots/confusion_matrix.png"

###############################################################
# HELPERS
###############################################################

def fastapi_status():

    try:

        response = requests.get(
            FASTAPI_URL,
            timeout=5
        )

        return response.status_code == 200

    except:

        return False


def mlflow_status():

    try:

        response = requests.get(
            MLFLOW_URL,
            timeout=5
        )

        return response.status_code == 200

    except:

        return False


###############################################################
# TITLE
###############################################################

st.title("⚽ Sports Betting MLOps Dashboard")

st.markdown(
"""
End-to-End Machine Learning Operations Project

**Technology Stack**

- Scikit-Learn
- FastAPI
- Streamlit
- MLflow
- Azure
"""
)

st.divider()

###############################################################
# SYSTEM HEALTH
###############################################################

st.subheader("🖥️ System Health")

c1, c2, c3, c4 = st.columns(4)

with c1:

    if fastapi_status():

        st.success("FastAPI Running")

    else:

        st.error("FastAPI Down")

with c2:

    if mlflow_status():

        st.success("MLflow Running")

    else:

        st.error("MLflow Down")

with c3:

    if os.path.exists(MODEL_PATH):

        st.success("Model Available")

    else:

        st.error("Model Missing")

with c4:

    if os.path.exists(PREPROCESSOR_PATH):

        st.success("Preprocessor Available")

    else:

        st.error("Preprocessor Missing")

st.divider()

###############################################################
# MODEL METRICS
###############################################################

st.subheader("📊 Latest Model Performance")

if os.path.exists(METRICS_PATH):

    with open(METRICS_PATH) as file:

        metrics = json.load(file)

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Best Model",
            metrics["best_model"]
        )

        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.4f}"
        )

    with m2:

        st.metric(
            "Precision",
            f"{metrics['precision']:.4f}"
        )

        st.metric(
            "Recall",
            f"{metrics['recall']:.4f}"
        )

    with m3:

        st.metric(
            "F1 Score",
            f"{metrics['f1_score']:.4f}"
        )

        st.metric(
            "ROC AUC",
            f"{metrics['roc_auc']:.4f}"
        )

else:

    st.warning("metrics.json not found.")

st.divider()

###############################################################
# QUICK ACTIONS
###############################################################

st.subheader("🚀 Quick Actions")

a1, a2 = st.columns(2)

with a1:

    st.link_button(

        "Open Swagger",

        f"{FASTAPI_URL}/docs"

    )

with a2:

    st.link_button(

        "Open MLflow",

        MLFLOW_URL

    )

st.info(
"""
Use the **Prediction** page from the left sidebar
to make predictions using the deployed model.
"""
)

st.divider()

###############################################################
# ARTIFACTS
###############################################################

st.subheader("📁 Latest Artifacts")

p1, p2 = st.columns(2)

with p1:

    if os.path.exists(CONFUSION_MATRIX):

        st.image(

            CONFUSION_MATRIX,

            caption="Confusion Matrix",

            use_container_width=True

        )

with p2:

    if os.path.exists(ROC_CURVE):

        st.image(

            ROC_CURVE,

            caption="ROC Curve",

            use_container_width=True

        )

st.divider()

###############################################################
# FOOTER
###############################################################

st.caption(
"""
Sports Betting MLOps Platform

FastAPI • Streamlit • MLflow • Azure
"""
)