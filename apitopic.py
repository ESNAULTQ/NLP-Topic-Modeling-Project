# api.py
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
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

class TextInput(BaseModel):
    text: str

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
