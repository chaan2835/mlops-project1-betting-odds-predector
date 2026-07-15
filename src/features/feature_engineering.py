import pandas as pd


def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["Odds_Difference"] = abs(
        df["Home_Team_Odds"]
        -
        df["Away_Team_Odds"]
    )

    df["Home_Probability"] = (
        1 / df["Home_Team_Odds"]
    )

    df["Away_Probability"] = (
        1 / df["Away_Team_Odds"]
    )

    return df