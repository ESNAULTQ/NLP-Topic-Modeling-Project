import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title='Analyse Exploratoire de Données', layout='wide')

# Titre et description de l'application
st.title("Topic modeling for Research papers")
st.write("""
Bienvenue dans cette application Streamlit dédiée à l'analyse exploratoire de textes scientifiques.
Vous pouvez télécharger un fichier CSV contenant des abstracts, et l'application effectuera plusieurs
étapes de prétraitement du texte, telles que le nettoyage, la suppression des stop words, la tokenisation,
le stemming et la lemmatisation. Vous pourrez ensuite visualiser un nuage de mots et un histogramme
de la distribution des tags dans vos données.
""")

# Fonctions de traitement du texte
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def tokenize_text(text):
    return word_tokenize(text)

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Fonction pour afficher le nuage de mots
def plot_wordcloud(texts, title):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=24)
    plt.axis('off')
    st.pyplot(plt)

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ABSTRACT' in df.columns:
        st.write("Données chargées avec succès.")

        # Prétraitement du texte
        df['cleaned_text'] = df['ABSTRACT'].apply(clean_text)
        st.write("Texte nettoyé.")

        df['text_without_stopwords'] = df['cleaned_text'].apply(remove_stopwords)
        st.write("Stop words supprimés.")

        df['tokens'] = df['text_without_stopwords'].apply(tokenize_text)
        st.write("Tokenisation terminée.")

        df['stemmed_text'] = df['tokens'].apply(stem_tokens)
        st.write("Stemming terminé.")

        df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)
        st.write("Lemmatisation terminée.")

        # Affichage du nuage de mots après lemmatisation
        st.header('Nuage de mots après lemmatisation')
        plot_wordcloud(df['lemmatized_tokens'].apply(lambda x: ' '.join(x)), 'Nuage de mots - Après Lemmatisation')

        # Génération de l'histogramme pour les colonnes disponibles
        st.header('Distribution des tags (Histogramme)')
        available_tags = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
        tag_distribution = 100 * (df[available_tags].sum() / df.shape[0])
        tag_distribution = tag_distribution.sort_values(ascending=False)

        plt.figure(figsize=(10, 5))
        tag_distribution.plot(kind='bar', width=0.5)
        plt.xticks(rotation=45)
        plt.title("Distribution des tags (Histogramme)")
        plt.xlabel('Tags')
        plt.ylabel('Pourcentage (%)')
        st.pyplot(plt)

    else:
        st.error("La colonne 'ABSTRACT' est requise dans le fichier CSV.")
