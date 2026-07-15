import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.config import PROCESSED_DATA_PATH


REQUIRED_COLUMNS = [
    "Match_ID",
    "Date",
    "Sport",
    "Home_Team",
    "Away_Team",
    "Home_Team_Odds",
    "Away_Team_Odds",
    "Draw_Odds",
    "Predicted_Winner",
    "Actual_Winner"
]


def validate_dataset():

    try:

        logger.info("Starting Data Validation")

        df = pd.read_csv(PROCESSED_DATA_PATH)

        # ---------------------------------------------------
        # Check Empty Dataset
        # ---------------------------------------------------

        if df.empty:
            raise Exception("Dataset is Empty")

        logger.info("Dataset is not empty")

        # ---------------------------------------------------
        # Required Columns
        # ---------------------------------------------------

        missing_columns = []

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:

            raise Exception(
                f"Missing Columns : {missing_columns}"
            )

        logger.info("All Required Columns Exist")

        # ---------------------------------------------------
        # Missing Values
        # ---------------------------------------------------

        missing = df.isnull().sum()

        print("\nMissing Values Before Imputation\n")
        print(missing)

        logger.info("Checking missing values")

        numeric_columns = [
            "Home_Team_Odds",
            "Away_Team_Odds",
            "Draw_Odds"
        ]

        for col in numeric_columns:

            if df[col].isnull().sum() > 0:

                mean_value = df[col].mean()

                df[col] = df[col].fillna(mean_value)

                logger.info(
                    f"{col} missing values filled with mean: {mean_value:.4f}"
                )

        missing_after = df.isnull().sum()

        print("\nMissing Values After Imputation\n")
        print(missing_after)

        logger.info("Missing value treatment completed")

        # ---------------------------------------------------
        # Odds Validation
        # ---------------------------------------------------

        odds_columns = [

            "Home_Team_Odds",

            "Away_Team_Odds",

            "Draw_Odds"

        ]

        for col in odds_columns:

            if (df[col] <= 0).any():

                raise Exception(
                    f"{col} contains invalid odds."
                )

        logger.info("Odds Validation Completed")

        print("\n")

        print("=" * 60)

        print("DATA VALIDATION SUCCESSFUL")

        print("=" * 60)

        return df

    except Exception as e:

        raise CustomException(e, sys)