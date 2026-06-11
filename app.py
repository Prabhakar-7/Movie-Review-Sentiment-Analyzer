import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):

    text = str(text)

    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter a Movie Review")

if st.button("Predict"):

    cleaned = clean_text(review)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    probs = model.predict_proba(vector)[0]

    st.success(f"Predicted Sentiment: {prediction.upper()}")

    st.write(f"Negative: {probs[0]*100:.2f}%")
    st.write(f"Positive: {probs[1]*100:.2f}%")