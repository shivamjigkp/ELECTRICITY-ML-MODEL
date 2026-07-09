import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw electricity demand CSV and splits it into a
        train set (up to 2023-12-31) and a test set (2024 onward),
        matching the split used in the notebook. Saves both to artifacts/.
        """
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("notebook/data/Electricity_Demand_Dataset.csv")
            logging.info("Read the raw dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            logging.info("Splitting data into train and test (date-based, time-series split)")
            train_set = df[df["Timestamp"] <= "2023-12-31"]
            test_set = df[df["Timestamp"] >= "2024-01-01"]

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
