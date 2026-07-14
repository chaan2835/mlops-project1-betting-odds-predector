import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

from src.logger import logger
from src.exception import CustomException

from src.config import (

    CLASSIFICATION_REPORT_PATH,

    FEATURE_IMPORTANCE_PATH,

    CONFUSION_MATRIX_PATH,

    ROC_CURVE_PATH

)

class ModelEvaluation:

    def __init__(self, artifact_dir="artifacts"):

        logger.info("Model Evaluation Initialized")

    ###########################################################
    # Classification Report
    ###########################################################

    def save_classification_report(
        self,
        y_test,
        prediction
    ):

        try:

            logger.info("Generating Classification Report")

            report = classification_report(
                y_test,
                prediction
            )

            os.makedirs(
                os.path.dirname(CLASSIFICATION_REPORT_PATH),
                exist_ok=True
            )

            report_path = CLASSIFICATION_REPORT_PATH

            with open(report_path, "w") as file:

                file.write(report)

            logger.info(
                f"Classification Report Saved : {report_path}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ###########################################################
    # Confusion Matrix
    ###########################################################

    def save_confusion_matrix(
        self,
        y_test,
        prediction
    ):

        try:

            logger.info("Generating Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                prediction
            )

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )

            disp.plot(cmap="Blues")
            os.makedirs(
                os.path.dirname(CONFUSION_MATRIX_PATH),
                exist_ok=True
            )

            image_path = CONFUSION_MATRIX_PATH

            plt.savefig(
                image_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"Confusion Matrix Saved : {image_path}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ###########################################################
    # ROC Curve
    ###########################################################

    def save_roc_curve(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            logger.info("Generating ROC Curve")

            probability = model.predict_proba(
                X_test
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                probability
            )

            auc_score = roc_auc_score(
                y_test,
                probability
            )

            plt.figure(figsize=(7, 6))

            plt.plot(
                fpr,
                tpr,
                label=f"AUC = {auc_score:.4f}"
            )

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            plt.xlabel("False Positive Rate")

            plt.ylabel("True Positive Rate")

            plt.title("ROC Curve")

            plt.legend()

            os.makedirs(
                os.path.dirname(ROC_CURVE_PATH),
                exist_ok=True
            )

            image_path = ROC_CURVE_PATH

            plt.savefig(
                image_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"ROC Curve Saved : {image_path}"
            )

        except Exception as e:

            raise CustomException(e, sys)

    ###########################################################
    # Feature Importance
    ###########################################################

    def save_feature_importance(
        self,
        model,
        preprocessor
    ):

        try:

            if not hasattr(model, "feature_importances_"):

                logger.warning(
                    "Selected model does not support Feature Importance"
                )

                return

            logger.info("Generating Feature Importance")

            importance = pd.DataFrame({

                "Feature":

                    preprocessor.get_feature_names_out(),

                "Importance":

                    model.feature_importances_

            })

            importance = importance.sort_values(

                by="Importance",

                ascending=False

            )

            os.makedirs(
                os.path.dirname(FEATURE_IMPORTANCE_PATH),
                exist_ok=True
            )

            csv_path = FEATURE_IMPORTANCE_PATH

            importance.to_csv(

                csv_path,

                index=False

            )

            logger.info(
                f"Feature Importance Saved : {csv_path}"
            )

        except Exception as e:

            raise CustomException(e, sys)