# apitopic.py
import json
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.db_utils import create_connection, insert_feedback

# Charger le modèle LDA et le vectorizer sauvegardés
lda_model = joblib.load('models/model_1/lda_model.pkl')
vectorizer = joblib.load('models/model_1/vectorizer.pkl')
with open('models/model_1/topics.json', 'r', encoding='utf-8') as file:
    # Charge le contenu du fichier JSON en tant que variable Python
    themes = json.load(file)



# Initialiser FastAPI
app = FastAPI()



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
