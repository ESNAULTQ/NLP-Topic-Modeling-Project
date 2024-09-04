# api.py
from fastapi import FastAPI
import numpy as np
import joblib
from model import lda_model

# Charger le modèle LDA et le vectorizer sauvegardés
lda_model = joblib.load('lda_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialiser FastAPI
app = FastAPI()

# Route pour faire une prédiction avec une requête GET
@app.get("/predict")
def predict(text: str):
    # Transformer le texte en vecteur à l'aide du vectorizer
    text_vector = vectorizer.transform([text])

    # Faire une prédiction avec le modèle LDA
    topic_distribution = lda_model.transform(text_vector)

    # Retourner la distribution des topics
    return {"topic_distribution": topic_distribution.tolist()}
