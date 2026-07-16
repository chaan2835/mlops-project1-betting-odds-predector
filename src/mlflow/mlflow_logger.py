import os
import sys

import mlflow
import mlflow.sklearn

from src.logger import logger
from src.exception import CustomException

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_RUN_NAME
)


class MLflowLogger:

    def __init__(self):

        try:

            logger.info("=" * 60)
            logger.info("INITIALIZING MLFLOW")
            logger.info("=" * 60)

            mlflow.set_tracking_uri(
                MLFLOW_TRACKING_URI
            )

            mlflow.set_experiment(
                MLFLOW_EXPERIMENT_NAME
            )

            logger.info(
                f"Tracking URI : {MLFLOW_TRACKING_URI}"
            )

            logger.info(
                f"Experiment : {MLFLOW_EXPERIMENT_NAME}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Enable Auto Logging
    ####################################################
    def enable_autolog(self):

        logger.info(
            "MLflow Autolog Disabled"
        )

        return    

    ####################################################
    # Start Run
    ####################################################

    def start_run(self, run_name):

        try:

            logger.info(
                f"Starting Run : {run_name}"
            )

            return mlflow.start_run(
                run_name=run_name
            )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Log Custom Artifacts
    ####################################################

    def log_artifacts(self, artifact_list):

        try:

            logger.info(
                "Logging Artifacts"
            )

            for artifact in artifact_list:

                if os.path.exists(artifact):

                    logger.info(f"Uploading : {artifact}")

                    mlflow.log_artifact(artifact)


            logger.info(
                "Artifacts Logged Successfully")

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Log Tags
    ####################################################

    def log_tags(self, tags):

        try:

            logger.info(
                "Logging Tags"
            )

            mlflow.set_tags(tags)

            logger.info(
                "Tags Logged Successfully"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Log Parameters
    ####################################################

    def log_parameters(self, parameters):

        """
        Optional.

        AutoLog already logs sklearn parameters.

        Use only for custom parameters.
        """

        try:

            if parameters:

                logger.info(
                    "Logging Custom Parameters"
                )

                mlflow.log_params(
                    parameters
                )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Log Metrics
    ####################################################

    def log_metrics(self, metrics):

        """
        Optional.

        AutoLog logs many metrics automatically.

        Use for project-specific metrics.
        """

        try:

            if metrics:

                logger.info(
                    "Logging Custom Metrics"
                )

                mlflow.log_metrics(
                    metrics
                )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # Register Model
    ####################################################

    def register_model(

        self,

        model,

        model_name

    ):

        """
        Optional.

        Register only the final best model.
        """

        try:

            logger.info(
                "Registering Model"
            )

            mlflow.sklearn.log_model(

                sk_model=model,

                artifact_path="model",

                registered_model_name=model_name

            )

            logger.info(
                "Model Registered Successfully"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ####################################################
    # End Run
    ####################################################

    def end_run(self):

        try:

            if mlflow.active_run():

                mlflow.end_run()

                logger.info(
                    "MLflow Run Ended Successfully"
                )

        except Exception as e:

            logger.error(
                f"Error Ending MLflow Run: {str(e)}"
            )
