import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DATA_PATH

def run_eda():

    # all your existing EDA code here
    
    EDA_DIR = "artifacts/eda"

    os.makedirs(
        EDA_DIR,
        exist_ok=True
    )

    ####################################################
    # Load Dataset
    ####################################################

    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("\nDataset Shape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    ####################################################
    # Target Variable
    ####################################################

    df["Prediction_Correct"] = (
        df["Predicted_Winner"]
        ==
        df["Actual_Winner"]
    ).astype(int)

    print("\nTarget Distribution")
    print(
        df["Prediction_Correct"].value_counts()
    )

    ####################################################
    # Target Distribution Plot
    ####################################################

    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="Prediction_Correct",
        data=df
    )

    plt.title(
        "Prediction Correct Distribution"
    )

    plt.savefig(
        f"{EDA_DIR}/target_distribution.png",
        bbox_inches="tight"
    )

    plt.close()

    ####################################################
    # Sport Distribution
    ####################################################

    plt.figure(figsize=(8, 5))

    sns.countplot(
        y="Sport",
        data=df,
        order=df["Sport"].value_counts().index
    )

    plt.title("Sport Distribution")

    plt.savefig(
        f"{EDA_DIR}/sport_distribution.png",
        bbox_inches="tight"
    )

    plt.close()

    ####################################################
    # Odds Distribution
    ####################################################

    odds_cols = [
        "Home_Team_Odds",
        "Away_Team_Odds",
        "Draw_Odds"
    ]

    for col in odds_cols:

        plt.figure(figsize=(7, 4))

        sns.histplot(
            df[col],
            kde=True,
            bins=30
        )

        plt.title(f"{col} Distribution")

        plt.savefig(
            f"{EDA_DIR}/{col}.png",
            bbox_inches="tight"
        )

        plt.close()

    ####################################################
    # Correlation Analysis
    ####################################################

    temp_df = df.copy()

    temp_df["Prediction_Correct"] = (
        temp_df["Predicted_Winner"]
        ==
        temp_df["Actual_Winner"]
    ).astype(int)

    corr_df = temp_df[
        [
            "Home_Team_Odds",
            "Away_Team_Odds",
            "Draw_Odds",
            "Prediction_Correct"
        ]
    ]

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        corr_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        f"{EDA_DIR}/correlation_heatmap.png",
        bbox_inches="tight"
    )

    plt.close()

    ####################################################
    # Prediction Accuracy by Sport
    ####################################################

    sport_accuracy = (
        df.groupby("Sport")["Prediction_Correct"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))

    sport_accuracy.plot(
        kind="bar"
    )

    plt.ylabel(
        "Prediction Correct Rate"
    )

    plt.title(
        "Prediction Accuracy by Sport"
    )

    plt.savefig(
        f"{EDA_DIR}/sport_accuracy.png",
        bbox_inches="tight"
    )

    plt.close()

    ####################################################
    # Summary Statistics
    ####################################################

    summary = df.describe(
        include="all"
    )

    summary.to_csv(
        f"{EDA_DIR}/summary_statistics.csv"
    )

    print("\nEDA Completed Successfully")

    print(
        f"\nArtifacts saved in: {EDA_DIR}"
    )
    print("EDA Completed Successfully")

if __name__ == "__main__":
    run_eda()