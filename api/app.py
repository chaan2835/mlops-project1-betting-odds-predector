from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException

import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(

    title="Sports Betting Prediction API",

    version="1.0",

    description="Production ML Model API for Sports Betting Prediction"

)


##########################################################
# Request Schema
##########################################################

class MatchData(BaseModel):

    Match_ID: str

    Date: str

    Sport: str

    Home_Team: str

    Away_Team: str

    Home_Team_Odds: float

    Away_Team_Odds: float

    Draw_Odds: float

    Predicted_Winner: str


##########################################################
# Health Check
##########################################################

@app.get("/")

def home():

    return {

        "status": "Running",

        "message": "Sports Betting Prediction API"

    }


##########################################################
# Prediction
##########################################################

@app.post("/predict")
def predict(data: MatchData):

    try:

        df = pd.DataFrame([{
            "Match_ID": data.Match_ID,
            "Date": data.Date,
            "Sport": data.Sport,
            "Home_Team": data.Home_Team,
            "Away_Team": data.Away_Team,
            "Home_Team_Odds": data.Home_Team_Odds,
            "Away_Team_Odds": data.Away_Team_Odds,
            "Draw_Odds": data.Draw_Odds,
            "Predicted_Winner": data.Predicted_Winner
        }])

        pipeline = PredictionPipeline()

        prediction, probability = pipeline.predict(df)

        return {
            "success": True,
            "Prediction_Correct": int(prediction[0]),
            "Probability": probability.tolist()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )