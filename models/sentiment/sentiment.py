import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import FastText
import numpy as np
import joblib

# Chargement des données
dftrain = pd.read_csv('NewWomensClothes.csv').dropna(subset=['Review Text', 'Rating'])

# Préparation des features et des labels
X = dftrain['Review Text']
y = dftrain['Rating'].apply(lambda rating: 'positive' if rating > 3 else 'negative')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle FastText
X_tokens_train = [text.split() for text in X_train]
fasttext_model = FastText(vector_size=10, window=5, min_count=2, epochs=10)
fasttext_model.build_vocab(X_tokens_train)
fasttext_model.train(X_tokens_train, total_examples=len(X_tokens_train), epochs=10)

# Fonction pour obtenir le vecteur moyen d'un document
def get_document_vector(text, model):
    tokens = text.split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Vectorisation des données d'entraînement et de test
X_train_vectors = np.array([get_document_vector(text, fasttext_model) for text in X_train])
X_test_vectors = np.array([get_document_vector(text, fasttext_model) for text in X_test])

# Entraînement du modèle de régression logistique
logistic_model = LogisticRegression(max_iter=100)
logistic_model.fit(X_train_vectors, y_train)

# Sauvegarde des modèles FastText et de régression logistique
joblib.dump(fasttext_model, 'fasttext_model.pkl')
joblib.dump(logistic_model, 'logistic_model.pkl')

# Évaluation sur l'ensemble de test
y_pred = logistic_model.predict(X_test_vectors)
print(classification_report(y_test, y_pred))

# Fonction de prédiction du sentiment
def predict_sentiment(text):
    vector = get_document_vector(text, fasttext_model)
    return logistic_model.predict([vector])[0]

# Exemple de prédiction
new_text = "The service was quite bad."
print(f"Sentiment: {predict_sentiment(new_text)}")
