import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException


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
        os.makedirs(os.path.join("artifacts", file_path), exist_ok=True)
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
