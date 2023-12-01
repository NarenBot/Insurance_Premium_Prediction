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

    def create_database(self):
        conn = sqlite3.connect(self.database_connect_config.database_path)
        cursor = conn.cursor()
        create_table = """CREATE TABLE IF NOT EXISTS user_details(
            id INTEGER PRIMARY KEY,
            name TEXT,
            age REAL,
            sex TEXT,
            bmi REAL,
            children REAL,
            smoker TEXT,
            region TEXT,
            expenses REAL);"""
        cursor.execute(create_table)
        conn.commit()
        return (conn, cursor)

    def insert_user_data(self, name, age, sex, bmi, children, smoker, region, expenses):
        conn, cursor = self.create_database()
        insert_values = """INSERT INTO user_details(
            name, age, sex, bmi, children, smoker, region, expenses) VALUES(
            ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(
            insert_values, (name, age, sex, bmi, children, smoker, region, expenses)
        )
        conn.commit()

    def display_user_database(self):
        conn, cursor = self.create_database()
        fetch = cursor.execute("SELECT * FROM user_details")
        data = fetch.fetchall()
        return (data, cursor)
