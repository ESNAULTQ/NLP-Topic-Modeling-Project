#Modèle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

dftrain = pd.read_csv("Train.csv")

X_train = dftrain['ABSTRACT']

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(X_train_vec)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
tf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, tf_feature_names, no_top_words)

joblib.dump(lda_model, 'lda_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Modèle et vectorizer enregistrés en tant que 'lda_model.pkl' et 'vectorizer.pkl'")
