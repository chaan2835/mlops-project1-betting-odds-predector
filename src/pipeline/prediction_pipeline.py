import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.features.feature_engineering import create_features
from src.utils import load_object

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH
)


class PredictionPipeline:

    def __init__(self):

        logger.info("Prediction Pipeline Initialized")

        self.model = load_object(MODEL_PATH)

        self.preprocessor = load_object(PREPROCESSOR_PATH)

    ##########################################################
    # Predict
    ##########################################################

    def predict(self, input_df):

        try:

            logger.info("Transforming Input Data")

            input_df["Date"] = pd.to_datetime(
            input_df["Date"]
            )

            input_df["Year"] = (
                input_df["Date"].dt.year
            )

            input_df["Month"] = (
                input_df["Date"].dt.month
            )

            input_df["DayOfWeek"] = (
                input_df["Date"].dt.dayofweek
            )

            input_df["Odds_Difference"] = abs(
                input_df["Home_Team_Odds"]
                -
                input_df["Away_Team_Odds"]
            )

            input_df["Home_Probability"] = (
                1 / input_df["Home_Team_Odds"]
            )

            input_df["Away_Probability"] = (
                1 / input_df["Away_Team_Odds"]
            )

            input_df = create_features(input_df)
            
            transformed = self.preprocessor.transform(input_df)

            logger.info("Predicting")

            prediction = self.model.predict(transformed)

            probability = self.model.predict_proba(transformed)

            return prediction, probability

        except Exception as e:

            raise CustomException(e, sys)