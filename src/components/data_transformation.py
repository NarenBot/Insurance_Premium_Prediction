import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    target_column = "expenses"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            data = pd.read_csv(os.path.join("artifacts", "data.csv"))
            num_columns = []
            cat_columns = []
            for col in data.columns:
                if data[col].dtype == "object":
                    cat_columns.append(col)
                else:
                    if self.data_transformation_config.target_column not in col:
                        num_columns.append(col)

            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns),
                ]
            )
            logging.info(num_columns)
            logging.info(cat_columns)

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Getting train_path and test_path...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_transformer_object()
            target_feature = self.data_transformation_config.target_column

            input_feature_train = train_df.drop(target_feature, axis=1)
            target_feature_train = train_df[target_feature]
            input_feature_test = test_df.drop(target_feature, axis=1)
            target_feature_test = test_df[target_feature]

            logging.info("All datas are going to scale...")
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train
            )
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test)
            logging.info(f"Scaled data sample: {input_feature_test_arr[0]}")
            logging.info(f"Number of features: {len(input_feature_test_arr[0])}")

            train_arr = np.c_[input_feature_train_arr, target_feature_train]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]

            logging.info("Saving the transformer object...")
            save_object(
                preprocessor_obj, self.data_transformation_config.preprocessor_path
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
