import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")

DATA_PATH = "data/raw/sports_betting_predictive_analysis.csv"

# Load once to populate dropdowns
df = pd.read_csv(DATA_PATH)

sports = sorted(df["Sport"].dropna().unique().tolist())

teams = sorted(
    list(
        set(df["Home_Team"].dropna().tolist())
        |
        set(df["Away_Team"].dropna().tolist())
    )
)

st.set_page_config(
    page_title="Prediction",
    layout="wide"
)

st.title("🔮 Sports Betting Prediction")

st.markdown(
    "Create a new match and let the trained ML model predict whether the selected winner is likely to be correct."
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    match_id = st.text_input(
        "Match ID",
        value="M10001"
    )

    match_date = st.date_input(
        "Match Date"
    )

    sport = st.selectbox(
        "Sport",
        sports
    )

    home_team = st.selectbox(
        "Home Team",
        teams
    )

with col2:

    away_team_options = [
        team
        for team in teams
        if team != home_team
    ]

    away_team = st.selectbox(
        "Away Team",
        away_team_options
    )

    predicted_winner = st.selectbox(
        "Predicted Winner",
        [home_team, away_team]
    )

st.divider()

st.subheader("Betting Odds")

c1, c2, c3 = st.columns(3)

with c1:

    home_odds = st.number_input(
        "Home Team Odds",
        min_value=1.01,
        max_value=20.00,
        value=2.00,
        step=0.01
    )

with c2:

    away_odds = st.number_input(
        "Away Team Odds",
        min_value=1.01,
        max_value=20.00,
        value=2.50,
        step=0.01
    )

with c3:

    draw_odds = st.number_input(
        "Draw Odds",
        min_value=1.01,
        max_value=20.00,
        value=3.00,
        step=0.01
    )

st.divider()

if st.button(
    "Predict",
    use_container_width=True,
    type="primary"
):

    payload = {

        "Match_ID": match_id,

        "Date": str(match_date),

        "Sport": sport,

        "Home_Team": home_team,

        "Away_Team": away_team,

        "Home_Team_Odds": float(home_odds),

        "Away_Team_Odds": float(away_odds),

        "Draw_Odds": float(draw_odds),

        "Predicted_Winner": predicted_winner

    }

    with st.spinner("Calling FastAPI..."):

        try:

            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json=payload,
                timeout=30
            )

            st.write("Status Code:", response.status_code)
            st.code(response.text)

            if response.status_code != 200:

                st.stop()

            st.write("Status Code:",response.status_code)

            st.code(response.text)    
            result = response.json()
     
            prediction = result["Prediction_Correct"]

            probability = result["Probability"][0][1]

            likely_winner = result["Likely_Winner"]


            st.success(
                "Prediction Completed"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Prediction",
                    "Correct ✅"
                    if prediction == 1
                    else "Incorrect ❌"
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{probability * 100:.2f}%"
                )
            
            with c3:

                st.metric(
                    "Likely Winner",
                    likely_winner
                )

            with st.expander(
                "Request Sent To FastAPI"
            ):

                st.json(payload)

            with st.expander(
                "Response"
            ):

                st.json(result)

        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )