import subprocess
import webbrowser
import time

import streamlit as st

st.set_page_config(
    page_title="Sports Betting MLOps Dashboard",
    layout="wide"
)

st.title("⚽ Sports Betting MLOps Dashboard")

st.markdown("---")

col1, col2, col3 = st.columns(3)

#########################################################
# FastAPI
#########################################################

with col1:

    st.subheader("FastAPI")

    if st.button("Start FastAPI"):

        subprocess.run(
            ["pkill", "-f", "uvicorn"],
            check=False
        )

        subprocess.Popen([
            "uvicorn",
            "api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000"
        ])

        time.sleep(5)

        api_url = (
            "https://chandrasekhardec20961-8000.eastus.instances.azureml.ms/docs"
        )

        st.success("FastAPI Started")

        st.link_button(
            "Open Swagger UI",
            api_url
        )

#########################################################
# MLflow
#########################################################

with col2:

    st.subheader("MLflow")

    if st.button("Start MLflow"):

        subprocess.run(
            ["pkill", "-f", "mlflow"],
            check=False
        )

        subprocess.Popen([
            "mlflow",
            "ui",
            "--host",
            "0.0.0.0",
            "--port",
            "5001"
        ])

        time.sleep(5)

        mlflow_url = (
            "https://chandrasekhardec20961-5001.eastus.instances.azureml.ms"
        )

        st.success("MLflow Started")

        st.link_button(
            "Open MLflow Dashboard",
            mlflow_url
        )

#########################################################
# Project Reports
#########################################################

with col3:

    st.subheader("Artifacts")

    if st.button("Open Artifacts Folder"):

        webbrowser.open("artifacts")
