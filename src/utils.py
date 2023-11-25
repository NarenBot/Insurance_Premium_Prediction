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
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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
        logging.info(f"Best parameters: {gs.best_params_}")

        best_model.set_params(**gs.best_params_)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        tuned_score = r2_score(y_test, y_pred)

        return tuned_score

    except Exception as e:
        raise CustomException(e, sys)
