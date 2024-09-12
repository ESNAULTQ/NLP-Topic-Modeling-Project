# apitopic.py
import json
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from utils.db_utils import create_connection, insert_feedback

# Charger le modèle LDA et le vectorizer sauvegardés
lda_model = joblib.load('models/model_1/lda_model.pkl')
vectorizer = joblib.load('models/model_1/vectorizer.pkl')
# Charger le modèle de sentiment analysis
fasttext_model = joblib.load('models/sentiment/fasttext_model.pkl')
logistic_model = joblib.load('models/sentiment/logistic_model.pkl')

with open('models/model_1/topics.json', 'r', encoding='utf-8') as file:
    # Charge le contenu du fichier JSON en tant que variable Python
    themes = json.load(file)

def get_document_vector(text):
    tokens = text.split()
    vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(fasttext_model.vector_size)

def predict_sentiment(text):
    vector = get_document_vector(text)
    return logistic_model.predict([vector])[0]


# Initialiser FastAPI
app = FastAPI()

class TextInput(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text_input: str
    predicted_value: str
    real_value: str

@app.get("/status")
def status():
    return {'message':'API en ligne!'}

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
        for i, prob in enumerate(topic_distribution)]

    return {"topic_distribution": topic_with_themes}


# Route pour faire une prédiction avec une requête POST
@app.post("/predict_s")
def predict_s(input_data: TextInput):
    text = input_data.text
    sentiment = predict_sentiment(text)
    return {"sentiment": sentiment}


@app.post("/feedback_topic")
def feedback_topic(feedback: FeedbackRequest):
    connection = create_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Erreur de connexion à la base de données.")

    insert_feedback("monitoring_topic", feedback.text_input, feedback.predicted_value, feedback.real_value, connection)

    connection.close()
    return {"message": "Feedback enregistré avec succès."}

@app.post("/feedback_sentiment")
def feedback_sentiment(feedback: FeedbackRequest):
    connection = create_connection()
    if connection is None:
        raise HTTPException(status_code=500, detail="Erreur de connexion à la base de données.")

    insert_feedback("monitoring_sentiment", feedback.text_input, feedback.predicted_value, feedback.real_value, connection)

    connection.close()
    return {"message": "Feedback enregistré avec succès."}
