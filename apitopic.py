# api.py
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
# Charger le modèle LDA et le vectorizer sauvegardés
lda_model = joblib.load('models/model_1/lda_model.pkl')
vectorizer = joblib.load('models/model_1/vectorizer.pkl')

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
    topic_distribution = lda_model.transform(text_vector)

    # Retourner la distribution des topics
    return {"topic_distribution": topic_distribution.tolist()}
@app.get("/status")
def status():
    return {'message':'API en ligne!'}
