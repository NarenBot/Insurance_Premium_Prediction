import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_best_model, finetune_best_model, mlflow_tracking


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train = train_arr[:, :-1]
            X_test = test_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_test = test_arr[:, -1]

            model_report, best_model_name, best_model, best_score = get_best_model(
                X_train, X_test, y_train, y_test
            )
            logging.info(model_report)
            logging.info(f"Best_Model: {best_model}, Best_Score: {best_score}")

            best_parameters, tuned_score = finetune_best_model(
                X_train, X_test, y_train, y_test, best_model_name, best_model
            )
            logging.info(f"Tuned_Model: {best_model}, Tuned_Score: {tuned_score}")

            tracking = mlflow_tracking(
                X_train, X_test, y_train, y_test, best_model, best_parameters
            )

            save_object(best_model, self.model_trainer_config.model_path)
            logging.info("Model object has saved.")

            return tuned_score

        except Exception as e:
            raise CustomException(e, sys)
