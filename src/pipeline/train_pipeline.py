import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def run(self):
        try:
            logging.info("Training pipeline started")

            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path, test_path
            )

            model_trainer = ModelTrainer()
            metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed. Metrics: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    print(pipeline.run())
