import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import holidays

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Same feature engineering steps as the notebook:
        datetime index, missing value handling, calendar features,
        lag features (24hr, 168hr), rolling mean/std (24hr window).
        """
        data = data.copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        data = data.set_index("Timestamp")

        data = data.dropna(how="all")
        data[["hour", "dayofweek", "month", "year", "dayofyear"]] = \
            data[["hour", "dayofweek", "month", "year", "dayofyear"]].ffill()
        data[["Temperature", "Humidity"]] = data[["Temperature", "Humidity"]].bfill()
        data["Demand"] = data["Demand"].interpolate(method="time")

        data.insert(5, "quarter", data.index.quarter)
        data[["hour", "dayofweek", "month", "year", "dayofyear"]] = \
            data[["hour", "dayofweek", "month", "year", "dayofyear"]].astype(int)
        data.insert(5, "weekofyear", data.index.isocalendar().week.astype(int))
        data.insert(7, "is_weekend", data.index.dayofweek.isin([5, 6]))
        data["is_weekend"] = data["is_weekend"].astype(int)

        # holiday calendar computed for parity with the notebook (not used as a
        # standalone feature — kept out of the final feature set)
        _ = holidays.IN(years=data.year)

        data["Demand_lag_24hr"] = data["Demand"].shift(24)
        data["demand_lag_168hr"] = data["Demand"].shift(168)
        data["demand_rolling_mean_24hr"] = data["Demand"].rolling(window=24).mean()
        data["demand_rolling_std_24hr"] = data["Demand"].rolling(window=24).std()

        return data

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test raw data")

            # Concatenate before feature engineering so that lag/rolling
            # features at the START of the test set can look back into the
            # END of the train set (avoids NaNs from lost history at the split
            # boundary), then split again by date after features are built.
            full_df = pd.concat([train_df, test_df], axis=0)
            full_features = self.build_features(full_df)
            full_features = full_features.dropna()

            train_features = full_features.loc[:"2023-12-31"]
            test_features = full_features.loc["2024-01-01":]

            logging.info("Feature engineering completed on train and test data")

            target_column = "Demand"

            input_feature_train_df = train_features.drop(columns=[target_column])
            target_feature_train_df = train_features[target_column]

            input_feature_test_df = test_features.drop(columns=[target_column])
            target_feature_test_df = test_features[target_column]

            # Tree-based model (XGBoost) doesn't require scaling, but we still
            # persist the feature column order as a "preprocessor" artifact so
            # the predict pipeline can align incoming data the same way every time.
            preprocessor = {"feature_columns": list(input_feature_train_df.columns)}

            train_arr = np.c_[
                input_feature_train_df[preprocessor["feature_columns"]].values,
                np.array(target_feature_train_df),
            ]
            test_arr = np.c_[
                input_feature_test_df[preprocessor["feature_columns"]].values,
                np.array(target_feature_test_df),
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
