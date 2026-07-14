import os
import sys
import pandas as pd

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH
)


@dataclass
class DataIngestionConfig:

    raw_data_path: str = RAW_DATA_PATH

    processed_data_path: str = PROCESSED_DATA_PATH


class DataIngestion:

    def __init__(self):

        self.config = DataIngestionConfig()

    ####################################################
    # Data Ingestion
    ####################################################

    def initiate_data_ingestion(self):

        logger.info("=" * 60)
        logger.info("DATA INGESTION STARTED")
        logger.info("=" * 60)

        try:

            ####################################################
            # Check Dataset
            ####################################################

            if not os.path.exists(self.config.raw_data_path):

                raise FileNotFoundError(
                    f"Dataset not found : {self.config.raw_data_path}"
                )

            logger.info(
                f"Dataset Found : {self.config.raw_data_path}"
            )

            ####################################################
            # Read Dataset
            ####################################################

            logger.info("Reading Dataset")

            df = pd.read_csv(
                self.config.raw_data_path
            )

            logger.info(
                f"Dataset Shape : {df.shape}"
            )

            ####################################################
            # Remove Duplicate Records
            ####################################################

            before = len(df)

            df.drop_duplicates(
                inplace=True
            )

            after = len(df)

            logger.info(
                f"Duplicate Records Removed : {before-after}"
            )

            ####################################################
            # Create Processed Directory
            ####################################################

            os.makedirs(

                os.path.dirname(
                    self.config.processed_data_path
                ),

                exist_ok=True

            )

            ####################################################
            # Save Cleaned Dataset
            ####################################################

            df.to_csv(

                self.config.processed_data_path,

                index=False

            )

            logger.info(
                f"Processed Dataset Saved : {self.config.processed_data_path}"
            )

            logger.info("=" * 60)
            logger.info("DATA INGESTION COMPLETED")
            logger.info("=" * 60)

            return df

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)