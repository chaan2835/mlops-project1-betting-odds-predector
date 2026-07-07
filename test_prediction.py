import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

# Read one real record
df = pd.read_csv("data/processed/cleaned_data.csv")

sample = df.iloc[[0]].drop(columns=["Actual_Winner"])

print(sample)

pipeline = PredictionPipeline()

prediction, probability = pipeline.predict(sample)

print("\nPrediction :", prediction)

print("\nProbability :")

print(probability)

print("\nExpected Winner :", df.iloc[0]["Actual_Winner"])

print("Predicted Winner:", df.iloc[0]["Predicted_Winner"])