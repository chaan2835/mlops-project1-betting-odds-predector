# import os
# import joblib


# def save_object(file_path, obj):
#     directory = os.path.dirname(file_path)

#     os.makedirs(directory, exist_ok=True)

#     joblib.dump(obj, file_path)


# def load_object(file_path):
#     return joblib.load(file_path)
import os
import sys
import json
import joblib


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score
)


from src.logger import logger
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any python object using joblib
    """

    try:

        directory = os.path.dirname(file_path)

        os.makedirs(directory, exist_ok=True)

        joblib.dump(obj, file_path)

        logger.info(f"Object saved successfully : {file_path}")

    except Exception as e:

        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load saved object
    """

    try:

        logger.info(f"Loading object : {file_path}")

        return joblib.load(file_path)

    except Exception as e:

        raise CustomException(e, sys)


def save_json(file_path, dictionary):
    """
    Save dictionary as json
    """

    try:

        directory = os.path.dirname(file_path)

        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:

            json.dump(
                dictionary,
                file,
                indent=4
            )

        logger.info("Metrics JSON Saved")

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_model(

        model,

        X_train,

        y_train,

        X_test,

        y_test

):

    try:

        logger.info(f"Training {model.__class__.__name__}")

        model.fit(

            X_train,

            y_train

        )

        prediction = model.predict(

            X_test

        )

        probability = model.predict_proba(

            X_test

        )[:,1]

        accuracy = accuracy_score(

            y_test,

            prediction

        )

        precision = precision_score(

            y_test,

            prediction,

            zero_division=0

        )

        recall = recall_score(

            y_test,

            prediction,

            zero_division=0

        )

        f1 = f1_score(

            y_test,

            prediction,

            zero_division=0

        )

        roc_auc = roc_auc_score(

            y_test,

            probability

        )

        report = {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "roc_auc": roc_auc

        }

        return report

    except Exception as e:

        raise CustomException(e,sys)