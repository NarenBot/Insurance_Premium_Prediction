import os
import sys
import pandas as pd
import numpy as np
import dill
import yaml

from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

import mlflow  # ML-Flow Tracking
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")


def missing_treatment(missing_df):
    columns = missing_df.columns
    for col in columns:
        if missing_df[col].dtype == "object":
            missing_df[col].fillna(missing_df[col].mode()[0], inplace=True)
        else:
            missing_df[col].fillna(missing_df[col].median(), inplace=True)

    return missing_df


def save_object(obj, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(obj=obj, file=file)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file=file)
    except Exception as e:
        raise CustomException(e, sys)


def load_yaml(file_path):
    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys)


def eval_metrics(actual, predicted):
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        return (rmse, mae, r2)
    except Exception as e:
        raise CustomException(e, sys)


def get_best_model(X_train, X_test, y_train, y_test):
    try:
        models = {
            "Linear_reg": LinearRegression(),
            "Decision_tree_reg": DecisionTreeRegressor(),
            "Random_forest_reg": RandomForestRegressor(),
            "Ada_boost_reg": AdaBoostRegressor(),
            "Gradient_boost_reg": GradientBoostingRegressor(),
        }

        model_report: dict = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_report[list(models)[i]] = score

        best_score = max(sorted(list(model_report.values())))
        best_model_name = list(model_report)[
            list(model_report.values()).index(best_score)
        ]
        best_model = models[best_model_name]

        return (model_report, best_model_name, best_model, best_score)

    except Exception as e:
        raise CustomException(e, sys)


def finetune_best_model(X_train, X_test, y_train, y_test, best_model_name, best_model):
    try:
        logging.info("Loading yaml file...")
        params = load_yaml(os.path.join("config", "params.yaml"))
        param_grid = params["models"][best_model_name]["param_grid"]
        logging.info(f"Param_Grid: {param_grid}")

        gs = GridSearchCV(best_model, param_grid=param_grid, cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_parameters = gs.best_params_
        logging.info(f"Best parameters: {best_parameters}")

        best_model.set_params(**gs.best_params_)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        tuned_score = r2_score(y_test, y_pred)

        return best_parameters, tuned_score

    except Exception as e:
        raise CustomException(e, sys)


def mlflow_tracking(X_train, X_test, y_train, y_test, best_model, best_parameters):
    try:
        with mlflow.start_run():
            logging.info("ML-Flow Tracking Started...")
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            rmse, mae, r2 = eval_metrics(y_test, y_pred)
            logging.info(f"Metrics are RMSE: {rmse} | MAE: {mae} | R2: {r2}")

            keys = list(best_parameters.keys())
            values = list(best_parameters.values())
            for i in range(len(best_parameters)):
                mlflow.log_param(f"{keys[i]}", values[i])

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            predictions = best_model.predict(X_train)
            signature = infer_signature(X_train, predictions)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name=str(type(best_model)).split(".")[-1][:-2],
                    signature=signature,
                )
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature)

    except Exception as e:
        raise CustomException(e, sys)
