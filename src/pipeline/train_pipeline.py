import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self) -> None:
        pass

    def start_data_ingestion(self):
        try:
            logging.info("Data ingestion started.")
            obj1 = DataIngestion()
            self.train_data, self.test_data = obj1.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        try:
            logging.info("Data transformation started.")
            obj2 = DataTransformation()
            self.train_arr, self.test_arr, _ = obj2.initiate_data_transformation(
                self.train_data, self.test_data
            )
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self):
        try:
            logging.info("Model trainer started.")
            obj3 = ModelTrainer()
            self.tuned_score = obj3.initiate_model_trainer(
                self.train_arr, self.test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            self.start_data_ingestion()
            self.start_data_transformation()
            self.start_model_trainer()
            logging.info("Model Training completed successfully.")
            return self.tuned_score

        except Exception as e:
            raise CustomException(e, sys)
