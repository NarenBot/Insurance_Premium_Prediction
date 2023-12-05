import os
import sys
import sqlite3
from dataclasses import dataclass
import pymongo


@dataclass
class DatabaseConnectConfig:
    database_path = os.path.join("artifacts", "insurance.db")


class DatabaseConnect:
    def __init__(self):
        self.database_connect_config = DatabaseConnectConfig()

    def create_mongo_database(self):
        self.username = "insurance"
        self.password = "root_insurance"
        self.dbname = "insurance_db"
        self.tabname = "insurance_tab"
        self.url = f"mongodb+srv://{self.username}:{self.password}@cluster0.plgldst.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(self.url)
        database = client[self.dbname]
        collection = database[self.tabname]
        return collection

    def create_lite_database(self):
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
        conn, cursor = self.create_lite_database()
        insert_values = """INSERT INTO user_details(
            name, age, sex, bmi, children, smoker, region, expenses) VALUES(
            ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(
            insert_values, (name, age, sex, bmi, children, smoker, region, expenses)
        )
        conn.commit()

        ## Insert MongoDB:
        collection = self.create_mongo_database()
        document = {
            "name": name,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "expenses": expenses,
        }
        collection.insert_one(document=document)

    def display_user_database(self):
        conn, cursor = self.create_lite_database()
        fetch = cursor.execute("SELECT * FROM user_details")
        data = fetch.fetchall()
        return (data, cursor)
