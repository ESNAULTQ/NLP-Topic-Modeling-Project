import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import gensim
from gensim.models import FastText
import numpy as np
import joblib

# Chargement des données
NewW = pd.read_csv('NewWomensClothes.csv')

# Filtrer les lignes sans "Review Text" ou "Rating"
dftrain = NewW.dropna(subset=['Review Text', 'Rating'])

# Préparation des features et des labels
X = dftrain['Review Text']
y = dftrain['Rating'].apply(lambda rating: 'positive' if rating > 3 else 'negative')

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création d'une classe de transformation personnalisée pour l'intégration FastText
class FastTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=10, window=5, min_count=2, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        X_tokens = [text.split() for text in X]
        self.model = FastText(vector_size=self.vector_size, window=self.window, min_count=self.min_count, epochs=self.epochs)
        self.model.build_vocab(X_tokens)
        self.model.train(X_tokens, total_examples=len(X_tokens), epochs=self.model.epochs)
        return self

    def transform(self, X):
        X_tokens = [text.split() for text in X]
        return np.array([self.get_document_vector(tokens) for tokens in X_tokens])

    def get_document_vector(self, tokens):
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

# Création de la pipeline
pipeline = Pipeline([
    ('fasttext_vectorizer', FastTextVectorizer(vector_size=10, window=5, min_count=2, epochs=10)),
    ('logistic_regression', LogisticRegression(max_iter=100))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Sauvegarde de la pipeline complète dans un seul fichier
joblib.dump(pipeline, 'sentiment_analysis_pipeline.pkl')

# Prédiction et évaluation sur l'ensemble de test
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Fonction de prédiction du sentiment
def predict_sentiment(text):
    return pipeline.predict([text])[0]

# Exemple de prédiction
new_text = "The service was quite bad."
print(f"Sentiment de la phrase : {predict_sentiment(new_text)}")
