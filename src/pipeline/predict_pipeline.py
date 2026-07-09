import sys
import os

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            feature_columns = preprocessor["feature_columns"]
            features = features[feature_columns]

            preds = model.predict(features)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Collects inputs from the web form (or an API call) and turns them into
    a single-row dataframe with the exact feature columns the model expects.

    Note: because the model's strongest features are Demand_lag_24hr and
    demand_lag_168hr (past actual demand), the user must supply those recent
    demand values — this model is built for day-ahead/batch forecasting, not
    zero-history prediction from weather alone.
    """

    def __init__(
        self,
        timestamp: str,
        temperature: float,
        humidity: float,
        demand_lag_24hr: float,
        demand_lag_168hr: float,
        demand_rolling_mean_24hr: float,
        demand_rolling_std_24hr: float,
    ):
        self.timestamp = timestamp
        self.temperature = temperature
        self.humidity = humidity
        self.demand_lag_24hr = demand_lag_24hr
        self.demand_lag_168hr = demand_lag_168hr
        self.demand_rolling_mean_24hr = demand_rolling_mean_24hr
        self.demand_rolling_std_24hr = demand_rolling_std_24hr

    def get_data_as_data_frame(self):
        try:
            ts = pd.to_datetime(self.timestamp)

            custom_data_input_dict = {
                "hour": [ts.hour],
                "dayofweek": [ts.dayofweek],
                "month": [ts.month],
                "year": [ts.year],
                "dayofyear": [ts.dayofyear],
                "weekofyear": [int(ts.isocalendar()[1])],
                "quarter": [ts.quarter],
                "is_weekend": [1 if ts.dayofweek in [5, 6] else 0],
                "Temperature": [self.temperature],
                "Humidity": [self.humidity],
                "Demand_lag_24hr": [self.demand_lag_24hr],
                "demand_lag_168hr": [self.demand_lag_168hr],
                "demand_rolling_mean_24hr": [self.demand_rolling_mean_24hr],
                "demand_rolling_std_24hr": [self.demand_rolling_std_24hr],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
