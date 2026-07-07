# checking the application is running or not
# from src.logger import logger

# logger.info("Application Started")

# print("=" * 50)
# print(" Betting Odds Prediction Model Started ")
# print("=" * 50)

# logger.info("Application Finished")

# checking the data ingestion code is working or not
# from src.logger import logger

# from src.components.data_ingestion import DataIngestion

# logger.info("Project Started")

# try:

#     ingestion = DataIngestion()

#     dataframe = ingestion.initiate_data_ingestion()

#     print("\n")

#     print("=" * 60)

#     print("DATA INGESTION COMPLETED")

#     print("=" * 60)

#     print("\nFirst Five Rows\n")

#     print(dataframe.head())

#     print("\n")

#     print("Dataset Shape :", dataframe.shape)

# except Exception as e:

#     print(e)

# logger.info("Project Finished")

# checking the data validation code is working or not
# from src.logger import logger

# from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import validate_dataset

# logger.info("Project Started")

# try:

#     ingestion = DataIngestion()

#     ingestion.initiate_data_ingestion()

#     df = validate_dataset()

#     print("\n")

#     print(df.head())

# except Exception as e:

#     print(e)

# logger.info("Project Finished")

