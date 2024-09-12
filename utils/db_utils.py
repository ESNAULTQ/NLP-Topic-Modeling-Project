# db_utils.py

import mysql.connector
from mysql.connector import Error
from datetime import datetime
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os


def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Erreur de connexion : {e}")
        return None


def insert_feedback(table_name, text_input, predicted_value, real_value, connection):
    try:
        cursor = connection.cursor()
        sql = f"""
        INSERT INTO {table_name} (text_input, predicted_topic, real_topic, prediction_time)
        VALUES (%s, %s, %s, %s)
        """
        values = (text_input, predicted_value, real_value, datetime.now())
        cursor.execute(sql, values)
        connection.commit()
        cursor.close()
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback : {e}")
