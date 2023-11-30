import os
import sys
import sqlite3
from dataclasses import dataclass


@dataclass
class DatabaseConnectConfig:
    database_path = os.path.join("artifacts", "insurance.db")


class DatabaseConnect:
    def __init__(self):
        self.database_connect_config = DatabaseConnectConfig()

    def get_data_from_database(self):
        pass
