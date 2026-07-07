from src.logger import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import validate_dataset
from src.components.model_trainer import ModelTrainer


def main():

    try:

        logger.info("=" * 70)
        logger.info("SPORTS BETTING MLOPS PROJECT STARTED")
        logger.info("=" * 70)

        ####################################################
        # Data Ingestion
        ####################################################

        logger.info("Step 1 : Data Ingestion")

        ingestion = DataIngestion()

        ingestion.initiate_data_ingestion()

        ####################################################
        # Data Validation
        ####################################################

        logger.info("Step 2 : Data Validation")

        validate_dataset()

        ####################################################
        # Model Training
        ####################################################

        logger.info("Step 3 : Model Training")

        trainer = ModelTrainer()

        best_model, preprocessor, metrics = trainer.initiate_model_training()

        ####################################################
        # Pipeline Completed
        ####################################################

        print("\n")

        print("=" * 70)

        print("PIPELINE EXECUTED SUCCESSFULLY")

        print("=" * 70)

        print(f"\nBest Model : {metrics['best_model']}")

        print(f"Accuracy   : {metrics['accuracy']:.4f}")

        print(f"Precision  : {metrics['precision']:.4f}")

        print(f"Recall     : {metrics['recall']:.4f}")

        print(f"F1 Score   : {metrics['f1_score']:.4f}")

        print("\nArtifacts Generated Successfully\n")

        print("artifacts/")

        print("├── data")

        print("│   ├── train.csv")

        print("│   └── test.csv")

        print("├── models")

        print("│   ├── model.pkl")

        print("│   └── preprocessor.pkl")

        print("├── reports")

        print("│   ├── metrics.json")

        print("│   ├── classification_report.txt")

        print("│   └── feature_importance.csv")

        print("└── plots")

        print("    ├── confusion_matrix.png")

        print("    └── roc_curve.png")

        logger.info("Pipeline Completed Successfully")

    except Exception as e:

        logger.error(str(e))

        print(e)


if __name__ == "__main__":

    main()