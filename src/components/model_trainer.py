import os
import sys

import mlflow
import pandas as pd

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from src.logger import logger
from src.exception import CustomException

from src.utils import (
    save_object,
    save_json,
    evaluate_model
)

from src.evaluation.model_evaluation import ModelEvaluation
from src.mlflow.mlflow_logger import MLflowLogger

from src.config import (

    PROCESSED_DATA_PATH,

    MODEL_PATH,

    PREPROCESSOR_PATH,

    METRICS_PATH,

    TRAIN_DATA_PATH,

    TEST_DATA_PATH,

    CLASSIFICATION_REPORT_PATH,

    FEATURE_IMPORTANCE_PATH,

    CONFUSION_MATRIX_PATH,

    ROC_CURVE_PATH,

    MLFLOW_RUN_NAME

)


class ModelTrainer:

    def __init__(self):

        logger.info("Model Trainer Initialized")

        self.mlflow_logger = MLflowLogger()

    ####################################################
    # Model Training
    ####################################################

    def initiate_model_training(self):

        try:

            logger.info("=" * 70)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 70)

            ####################################################
            # Read Dataset
            ####################################################

            df = pd.read_csv(
                PROCESSED_DATA_PATH
            )

            logger.info(
                f"Dataset Shape : {df.shape}"
            )

            ####################################################
            # Create Target
            ####################################################

            logger.info(
                "Creating Prediction_Correct"
            )

            df["Prediction_Correct"] = (

                df["Predicted_Winner"]

                ==

                df["Actual_Winner"]

            ).astype(int)

            df["Date"] = pd.to_datetime(df["Date"])

            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["DayOfWeek"] = df["Date"].dt.dayofweek

            df["Odds_Difference"] = abs(
                df["Home_Team_Odds"]
                -
                df["Away_Team_Odds"]
            )

            df["Home_Probability"] = (
                1 / df["Home_Team_Odds"]
            )

            df["Away_Probability"] = (
                1 / df["Away_Team_Odds"]
            )

            print("\nTarget Distribution\n")

            print(
                df["Prediction_Correct"].value_counts()
            )

            print("\nPercentage\n")

            print(
                df["Prediction_Correct"]
                .value_counts(normalize=True)
            )

            ####################################################
            # Features / Target
            ####################################################

            X = df.drop(

                columns=[

                    "Prediction_Correct",

                    "Actual_Winner"

                ]

            )

            y = df["Prediction_Correct"]

            ####################################################
            # Train Test Split
            ####################################################

            X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=y

            )

            ####################################################
            # Save Train/Test CSV
            ####################################################

            os.makedirs(

                os.path.dirname(
                    TRAIN_DATA_PATH
                ),

                exist_ok=True

            )

            train_df = X_train.copy()

            train_df["Prediction_Correct"] = y_train.values

            train_df.to_csv(

                TRAIN_DATA_PATH,

                index=False

            )

            test_df = X_test.copy()

            test_df["Prediction_Correct"] = y_test.values

            test_df.to_csv(

                TEST_DATA_PATH,

                index=False

            )

            ####################################################
            # Feature Lists
            ####################################################

            numerical_columns = [
                "Home_Team_Odds",
                "Away_Team_Odds",
                "Draw_Odds",
                "Year",
                "Month",
                "DayOfWeek",
                "Odds_Difference",
                "Home_Probability",
                "Away_Probability"
            ]

            categorical_columns = [
                "Sport",
                "Home_Team",
                "Away_Team",
                "Predicted_Winner"
            ]
                

            ####################################################
            # Numeric Pipeline
            ####################################################

            numeric_pipeline = Pipeline(

                steps=[

                    (

                        "imputer",

                        SimpleImputer(
                            strategy="mean"
                        )

                    ),

                    (

                        "scaler",

                        StandardScaler()

                    )

                ]

            )

            ####################################################
            # Categorical Pipeline
            ####################################################

            categorical_pipeline = Pipeline(

                steps=[

                    (

                        "imputer",

                        SimpleImputer(
                            strategy="most_frequent"
                        )

                    ),

                    (

                        "encoder",

                        OneHotEncoder(
                            handle_unknown="ignore"
                        )

                    )

                ]

            )

            ####################################################
            # Preprocessor
            ####################################################

            preprocessor = ColumnTransformer(

                transformers=[

                    (

                        "numeric",

                        numeric_pipeline,

                        numerical_columns

                    ),

                    (

                        "categorical",

                        categorical_pipeline,

                        categorical_columns

                    )

                ]

            )

            ####################################################
            # Transform Data
            ####################################################

            X_train = preprocessor.fit_transform(
                X_train
            )

            X_test = preprocessor.transform(
                X_test
            )

            ####################################################
            # Save Preprocessor
            ####################################################

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            preprocessor_version_path = os.path.join(
                os.path.dirname(PREPROCESSOR_PATH),
                f"preprocessor_{timestamp}.pkl"
            )

            save_object(
                preprocessor_version_path,
                preprocessor
            )

            save_object(
                PREPROCESSOR_PATH,
                preprocessor
            )

            ####################################################
            # Initialize MLflow
            ####################################################
   
            mlflow.start_run(
                run_name=MLFLOW_RUN_NAME
            )

            ####################################################
            # Models
            ####################################################

            models = {

                "Logistic Regression": LogisticRegression(

                    max_iter=1000,

                    random_state=42

                ),

                "Decision Tree": DecisionTreeClassifier(

                    max_depth=10,

                    min_samples_split=5,

                    random_state=42

                ),

                "Random Forest": RandomForestClassifier(

                    n_estimators=300,

                    max_depth=15,

                    min_samples_split=5,

                    min_samples_leaf=2,

                    random_state=42,

                    n_jobs=-1

                ),

                "Gradient Boosting": GradientBoostingClassifier(

                    n_estimators=200,

                    learning_rate=0.05,

                    max_depth=5,

                    random_state=42

                ),

                "XGBoost": XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42
                ),

                "Extra Trees": ExtraTreesClassifier(

                    n_estimators=300,

                    random_state=42,

                    n_jobs=-1

                )

            }

            logger.info("Models Initialized Successfully")

            ####################################################
            # Model Training
            ####################################################

            model_report = {}

            best_model = None

            best_model_name = None

            best_f1 = -1

            logger.info("=" * 60)

            logger.info("Training All Models")

            logger.info("=" * 60)

            for model_name, model in models.items():

                logger.info(f"Training : {model_name}")

                report = evaluate_model(

                    model,

                    X_train,

                    y_train,

                    X_test,

                    y_test

                )

                model_report[model_name] = report

                logger.info(report)

                if report["f1_score"] > best_f1:

                    best_f1 = report["f1_score"]

                    best_model = model

                    best_model_name = model_name

            ####################################################
            # Best Model
            ####################################################

            logger.info("=" * 60)

            logger.info(f"Best Model : {best_model_name}")

            logger.info(f"Best F1    : {best_f1:.4f}")

            logger.info("=" * 60)


            ####################################################
            # Versioned Model Save
            ####################################################

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            model_version_path = os.path.join(
                os.path.dirname(MODEL_PATH),
                f"{best_model_name.replace(' ', '_')}_{timestamp}.pkl"
            )

            # Save versioned model
            save_object(
                model_version_path,
                best_model
            )

            # Save latest model for API consumption
            save_object(
                MODEL_PATH,
                best_model
            )

            logger.info(
                f"Versioned Model Saved : {model_version_path}"
            )

            logger.info(
                f"Latest Model Saved : {MODEL_PATH}"
            )

            ####################################################
            # Model Evaluation
            ####################################################

            logger.info("Evaluating Best Model")

            prediction = best_model.predict(

                X_test

            )

            evaluator = ModelEvaluation()

            evaluator.save_classification_report(

                y_test,

                prediction

            )

            evaluator.save_confusion_matrix(

                y_test,

                prediction

            )

            evaluator.save_roc_curve(

                best_model,

                X_test,

                y_test

            )

            evaluator.save_feature_importance(

                best_model,

                preprocessor

            )

            logger.info(

                "Model Evaluation Completed"

            )

            ####################################################
            # Log Custom Artifacts
            ####################################################

            artifact_files = [

                TRAIN_DATA_PATH,

                TEST_DATA_PATH,

                METRICS_PATH,

                CLASSIFICATION_REPORT_PATH,

                FEATURE_IMPORTANCE_PATH,

                CONFUSION_MATRIX_PATH,

                ROC_CURVE_PATH

            ]

            self.mlflow_logger.log_artifacts(

                artifact_files

            )

            ####################################################
            # Log Tags
            ####################################################

            self.mlflow_logger.log_tags({

                "Project": "Sports Betting MLOps",

                "Developer": "Chandra Sekhar",

                "Environment": "Azure",

                "Algorithm": best_model_name,

                "Framework": "Scikit-Learn"

            })

            ####################################################
            # Register Best Model
            ####################################################

            # self.mlflow_logger.register_model(

            #     best_model,

            #     "Sports_Betting_Model"

            # )

            ####################################################
            # Metrics for metrics.json
            ####################################################

            metrics = {
                "best_model": best_model_name,
                "accuracy": round(
                    model_report[best_model_name]["accuracy"],
                    4
                ),
                "precision": round(
                    model_report[best_model_name]["precision"],
                    4
                ),
                "recall": round(
                    model_report[best_model_name]["recall"],
                    4
                ),
                "f1_score": round(
                    model_report[best_model_name]["f1_score"],
                    4
                ),
                "roc_auc": round(
                    model_report[best_model_name]["roc_auc"],
                    4
                )
            }

            ####################################################
            # Save metrics.json
            ####################################################

            save_json(

                METRICS_PATH,

                metrics

            )

            logger.info(

                "metrics.json Saved Successfully"

            )

            ####################################################
            # Log metrics.json to MLflow
            ####################################################

            self.mlflow_logger.log_artifacts(

                [

                    METRICS_PATH

                ]

            )

            ####################################################
            # MLflow Metrics (Numeric Only)
            ####################################################

            mlflow_metrics = {

                "accuracy": metrics["accuracy"],

                "precision": metrics["precision"],

                "recall": metrics["recall"],

                "f1_score": metrics["f1_score"],

                "roc_auc": metrics["roc_auc"]
            }

            mlflow_metrics = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"]
            }

            self.mlflow_logger.log_metrics(
                mlflow_metrics
            )


            ####################################################
            # Log Custom Parameters
            ####################################################

            self.mlflow_logger.log_parameters(

                {

                    "Best_Model": best_model_name,

                    "Training_Samples": len(y_train),

                    "Testing_Samples": len(y_test),

                    "Numerical_Features": len(numerical_columns),

                    "Categorical_Features": len(categorical_columns)

                }

            )

            ####################################################
            # Close MLflow Run
            ####################################################

            if mlflow.active_run():
                mlflow.end_run()

            ####################################################
            # Console Output
            ####################################################

            print("\n")

            print("=" * 70)

            print("MODEL TRAINING COMPLETED")

            print("=" * 70)

            print(f"\nBest Model : {best_model_name}")

            print(f"Accuracy   : {metrics['accuracy']:.4f}")

            print(f"Precision  : {metrics['precision']:.4f}")

            print(f"Recall     : {metrics['recall']:.4f}")

            print(f"F1 Score   : {metrics['f1_score']:.4f}")

            print(f"ROC AUC    : {metrics['roc_auc']:.4f}")

            print("\nDetailed Report\n")

            for name, report in model_report.items():

                print(name)

                print(report)

                print("-" * 60)

            print("\n")

            print("=" * 70)

            print("MLFLOW EXPERIMENT LOGGED SUCCESSFULLY")

            print("=" * 70)

            print("Artifacts")

            print("✔ train.csv")

            print("✔ test.csv")

            print("✔ model.pkl")

            print("✔ preprocessor.pkl")

            print("✔ metrics.json")

            print("✔ classification_report.txt")

            print("✔ feature_importance.csv")

            print("✔ confusion_matrix.png")

            print("✔ roc_curve.png")

            ####################################################
            # Return
            ####################################################

            return (

                best_model,

                preprocessor,

                metrics

            )

        except Exception as e:

            logger.error(str(e))

            if mlflow.active_run():

                mlflow.end_run()

            raise CustomException(e, sys)