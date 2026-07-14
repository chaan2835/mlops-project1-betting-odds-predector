import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.evaluation.model_evaluation import ModelEvaluation

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
)


class ModelTrainer:

    def __init__(self):

        logger.info("Model Trainer Initialized")


    def initiate_model_training(self):

        try:

            logger.info("=" * 60)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 60)

            ####################################################
            # Read Dataset
            ####################################################

            df = pd.read_csv(PROCESSED_DATA_PATH)

            logger.info(f"Dataset Shape : {df.shape}")

            ####################################################
            # Create Target Column
            ####################################################

            logger.info("Creating Prediction_Correct")

            df["Prediction_Correct"] = (
                df["Predicted_Winner"] ==
                df["Actual_Winner"]
            ).astype(int)

            print("\nTarget Distribution\n")

            print(df["Prediction_Correct"].value_counts())

            print("\nPercentage\n")

            print(df["Prediction_Correct"].value_counts(normalize=True))

            logger.info("Target Created Successfully")

            logger.info(
                f"\nTarget Distribution\n"
                f"{df['Prediction_Correct'].value_counts()}"
            )

            ####################################################
            # Features
            ####################################################

            X = df.drop(
                columns=[
                    "Prediction_Correct",
                    "Actual_Winner"
                ]
            )

            ####################################################
            # Target
            ####################################################

            y = df["Prediction_Correct"]

            ####################################################
            # Split
            ####################################################

            X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=y

            )

            logger.info("Train Test Split Completed")

            ####################################################
            # Numeric Columns
            ####################################################

            numerical_columns = [

                "Home_Team_Odds",

                "Away_Team_Odds",

                "Draw_Odds"

            ]

            ####################################################
            # Categorical Columns
            ####################################################

            categorical_columns = [

                "Match_ID",

                "Date",

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
            # Combine Pipelines
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

            logger.info("Preprocessor Created")

            ####################################################
            # Fit
            ####################################################

            X_train = preprocessor.fit_transform(
                X_train
            )

            X_test = preprocessor.transform(
                X_test
            )

            logger.info("Preprocessing Completed")

            ####################################################
            # Save Preprocessor
            ####################################################

            save_object(

                PREPROCESSOR_PATH,

                preprocessor

            )

            logger.info(
                "Preprocessor Saved Successfully"
            )

            ####################################################
            # Models
            ####################################################

            models = {

                "Logistic Regression":

                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    ),

                "Decision Tree":

                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42
                    ),

                "Random Forest":

                    RandomForestClassifier(

                        n_estimators=300,

                        max_depth=15,

                        min_samples_split=5,

                        min_samples_leaf=2,

                        random_state=42,

                        n_jobs=-1

                    ),

                "Gradient Boosting":

                    GradientBoostingClassifier(

                        n_estimators=200,

                        learning_rate=0.05,

                        max_depth=5,

                        random_state=42

                    ),

                "Extra Trees":

                    ExtraTreesClassifier(

                        n_estimators=300,

                        random_state=42,

                        n_jobs=-1

                    )

            }
            logger.info(
                "Models Initialized Successfully"
            )

                        ####################################################
            # Train Models
            ####################################################

            model_report = {}

            best_f1 = 0.0

            best_model = None

            best_model_name = ""

            logger.info("=" * 60)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 60)

            for model_name, model in models.items():

                logger.info(f"Training {model_name}")

                report = evaluate_model(

                    model,

                    X_train,

                    y_train,

                    X_test,

                    y_test

                )

                model_report[model_name] = report

                logger.info(f"{model_name} Results")

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

            logger.info(f"F1 Score   : {best_f1}")

            logger.info("=" * 60)

            ####################################################
            # Save Best Model
            ####################################################

            save_object(

                MODEL_PATH,

                best_model

            )

            logger.info("Best Model Saved Successfully")

            ####################################################
            # Model Evaluation
            ####################################################

            logger.info("Starting Model Evaluation")

            evaluation = ModelEvaluation()

            prediction = best_model.predict(X_test)

            evaluation.save_classification_report(
                y_test,
                prediction
            )

            evaluation.save_confusion_matrix(
                y_test,
                prediction
            )

            evaluation.save_roc_curve(
                best_model,
                X_test,
                y_test
            )

            evaluation.save_feature_importance(
                best_model,
                preprocessor
            )

            logger.info("Model Evaluation Completed")

            ####################################################
            # Save Metrics
            ####################################################

            metrics = {

                "best_model": best_model_name,

                "accuracy": round(model_report[best_model_name]["accuracy"], 4),

                "precision": round(model_report[best_model_name]["precision"], 4),

                "recall": round(model_report[best_model_name]["recall"], 4),

                "f1_score": round(model_report[best_model_name]["f1_score"], 4),

                "roc_auc": round(model_report[best_model_name]["roc_auc"], 4)

            }

            save_json(

                METRICS_PATH,

                metrics

            )

            logger.info("Metrics Saved Successfully")

            ####################################################
            # Display Results
            ####################################################

            print("\n")

            print("=" * 60)

            print("MODEL TRAINING COMPLETED")

            print("=" * 60)

            print(f"Best Model : {best_model_name}")

            print("\n")

            print(f"Accuracy  : {metrics['accuracy']:.4f}")

            print(f"Precision : {metrics['precision']:.4f}")

            print(f"Recall    : {metrics['recall']:.4f}")

            print(f"F1 Score  : {metrics['f1_score']:.4f}")

            print(f"ROC AUC   : {metrics['roc_auc']:.4f}")

            print("\nDetailed Report\n")

            for name, report in model_report.items():

                print(name)

                print(report)

                print("-" * 50)

                print("\n")

            print("=" * 60)

            print("Artifacts Generated")

            print("=" * 60)

            print("✔ model.pkl")

            print("✔ preprocessor.pkl")

            print("✔ metrics.json")

            print("✔ classification_report.txt")

            print("✔ confusion_matrix.png")

            print("✔ roc_curve.png")

            print("✔ feature_importance.csv")

            ####################################################
            # Return
            ####################################################

            return (

                best_model,

                preprocessor,

                metrics

            )

        except Exception as e:

            raise CustomException(e, sys)