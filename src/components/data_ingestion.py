import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.utils import missing_treatment


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True
            )

            logging.info("Getting the source data...")
            df = pd.read_csv(r"research/insurance.csv")

            df = missing_treatment(df)
            logging.info(f"Treatment for any missing data.\n{df.isnull().sum()}")
            df.to_csv(
                self.data_ingestion_config.raw_data_path, header=True, index=False
            )

            logging.info("Splitting the data into train and test...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(
                self.data_ingestion_config.train_data_path, header=True, index=False
            )
            test_df.to_csv(
                self.data_ingestion_config.test_data_path, header=True, index=False
            )

            logging.info("All data files are saved.")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
