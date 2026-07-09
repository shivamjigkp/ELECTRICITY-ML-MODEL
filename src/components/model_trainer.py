import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model = XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=50,
                learning_rate=0.01,
                random_state=42,
                objective="reg:squarederror",
            )

            logging.info("Training XGBoost model")
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )

            predictions = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            logging.info(f"Model evaluation - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

            if r2 < 0.6:
                raise CustomException("No suitable model found (R2 below threshold)", sys)

            logging.info("Saving trained model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            return {"rmse": rmse, "mae": mae, "r2": r2}

        except Exception as e:
            raise CustomException(e, sys)
