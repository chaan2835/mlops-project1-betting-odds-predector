import os
from dotenv import load_dotenv

load_dotenv()

####################################################
# MLFLOW
####################################################
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME"
)
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME"
)

MLFLOW_RUN_NAME = os.getenv(
    "MLFLOW_RUN_NAME"
)

####################################################
# PROJECT ROOT
####################################################

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

####################################################
# DATA
####################################################

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATA_PATH = os.path.join(
    DATA_DIR,
    "raw",
    "sports_betting_predictive_analysis.csv"
)

PROCESSED_DATA_PATH = os.path.join(
    DATA_DIR,
    "processed",
    "cleaned_data.csv"
)

####################################################
# ARTIFACTS
####################################################

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

####################################################
# ARTIFACT DATA
####################################################

ARTIFACT_DATA_DIR = os.path.join(
    ARTIFACTS_DIR,
    "data"
)

TRAIN_DATA_PATH = os.path.join(
    ARTIFACT_DATA_DIR,
    "train.csv"
)

TEST_DATA_PATH = os.path.join(
    ARTIFACT_DATA_DIR,
    "test.csv"
)

####################################################
# MODELS
####################################################

MODELS_DIR = os.path.join(
    ARTIFACTS_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "preprocessor.pkl"
)

####################################################
# REPORTS
####################################################

REPORTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    "reports"
)

METRICS_PATH = os.path.join(
    REPORTS_DIR,
    "metrics.json"
)

CLASSIFICATION_REPORT_PATH = os.path.join(
    REPORTS_DIR,
    "classification_report.txt"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    REPORTS_DIR,
    "feature_importance.csv"
)

####################################################
# PLOTS
####################################################

PLOTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    "plots"
)

CONFUSION_MATRIX_PATH = os.path.join(
    PLOTS_DIR,
    "confusion_matrix.png"
)

ROC_CURVE_PATH = os.path.join(
    PLOTS_DIR,
    "roc_curve.png"
)