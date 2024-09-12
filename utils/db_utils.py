# db_utils.py

import mysql.connector
from mysql.connector import Error
from datetime import datetime
from fastapi import FastAPI, HTTPException

def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="your_database"
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
