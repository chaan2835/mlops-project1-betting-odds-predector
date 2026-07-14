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

        print("\nMissing Values\n")
        print(missing)

        logger.info("Checking missing values")

        missing_columns = missing[missing > 0]

        if len(missing_columns) > 0:

            logger.warning(
                f"Missing values found:\n{missing_columns}"
            )

            print("\nWARNING: Missing Values Found\n")

            print(missing_columns)

        else:

            logger.info("No Missing Values Found")

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