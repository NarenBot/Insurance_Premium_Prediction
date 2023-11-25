import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def prediction(self, dataframe):
        try:
            logging.info("Loading Preprocessor and Model files...")
            preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
            model = load_object(os.path.join("artifacts", "model.pkl"))
            scaled_data = preprocessor.transform(dataframe)
            prediction = model.predict(scaled_data)
            logging.info("Finally Predicted!")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, age, sex, bmi, children, smoker, region):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_dataframe(self):
        try:
            input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
            }
            logging.info("Creating Dataframe...")
            return pd.DataFrame(data=input_dict)

        except Exception as e:
            raise CustomException(e, sys)
