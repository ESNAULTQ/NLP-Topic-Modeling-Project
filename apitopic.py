# apitopic.py
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import os

# Charger le modèle LDA et le vectorizer sauvegardés
lda_model = joblib.load('models/model_1/lda_model.pkl')
vectorizer = joblib.load('models/model_1/vectorizer.pkl')

# Dictionnaire des thèmes associés aux topics
themes = {
    0: "Dynamique des Fluides et Énergétique",
    1: "Physique Quantique et Magnétisme",
    2: "Apprentissage Automatique et Intelligence Artificielle",
    3: "Réseaux Neuronaux et Apprentissage Profond",
    4: "Algorithmes et Théorie de la Complexité",
    5: "Astrophysique et Formation des Galaxies",
    6: "Cosmologie et Observation Radio",
    7: "Modélisation Mathématique et Méthodes Approximatives",
    8: "Théorie de l'Information et Communication",
    9: "Analyse des Réseaux Sociaux et des Données"
}

# Initialiser FastAPI
app = FastAPI()

def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return connection
    except Error as e:
        print(f"Erreur lors de la connexion à MySQL : {e}")
        return None

class TextInput(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text_input: str
    predicted_value: str
    real_value: str

# Route pour faire une prédiction avec une requête GET
@app.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text

    # Vérifier si le texte est vide
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte fourni est vide.")

    # Transformer le texte en vecteur à l'aide du vectorizer
    text_vector = vectorizer.transform([text])

    # Faire une prédiction avec le modèle LDA
    topic_distribution = lda_model.transform(text_vector)[0]

    # Associer les thèmes aux probabilités des topics
    topic_with_themes = [
        {"theme": themes[i], "probability": prob}
        for i, prob in enumerate(topic_distribution)
    ]

    # Retourner la distribution des topics avec les thèmes associés
    return {"topic_distribution": topic_with_themes}
@app.get("/status")
def status():
    return {'message':'API en ligne!'}

@app.post("/feedback_topic")
def feedback(feedback: FeedbackRequest):
    # Connexion à la base de données
    connection = create_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Erreur de connexion à la base de données.")

    try:
        cursor = connection.cursor()

        # Requête SQL pour insérer les données de feedback dans la table 'monitoring'
        sql = """
        INSERT INTO monitoring_topic (text_input, predicted_topic, real_topic, prediction_time)
        VALUES (%s, %s, %s, %s)
        """
        values = (

            feedback.text_input,
            feedback.predicted_value,
            feedback.real_value,
            datetime.now()
        )

        cursor.execute(sql, values)
        connection.commit()

        # Fermer le curseur et la connexion
        cursor.close()
        connection.close()

        return {"message": "Feedback enregistré avec succès."}
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback : {e}")

@app.post("/feedback_sentiment")

def feedback(feedback: FeedbackRequest):
    # Connexion à la base de données
    connection = create_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Erreur de connexion à la base de données.")

    try:
        cursor = connection.cursor()

        # Requête SQL pour insérer les données de feedback dans la table 'monitoring'
        sql = """
        INSERT INTO monitoring_sentiment (text_input, prediction, real_sentiment, prediction_time)
        VALUES (%s, %s, %s, %s)
        """
        values = (

            feedback.text_input,
            feedback.predicted_value,
            feedback.real_value,
            datetime.now()
        )

        cursor.execute(sql, values)
        connection.commit()

        # Fermer le curseur et la connexion
        cursor.close()
        connection.close()

        return {"message": "Feedback enregistré avec succès."}
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback : {e}")
