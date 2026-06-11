import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')

# Page Config
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: #0f172a;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 25px;
}

.result-positive {
    background-color: #dcfce7;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
    color: #166534;
    font-size: 20px;
    font-weight: bold;
}

.result-negative {
    background-color: #fee2e2;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #ef4444;
    color: #991b1b;
    font-size: 20px;
    font-weight: bold;
}

.stButton button {
    width: 100%;
    height: 3em;
    font-size: 16px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Load Model and Vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Stopwords loaded once
stop_words = set(stopwords.words('english'))

# Text Cleaning Function
def clean_text(text):

    text = str(text)

    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()

    words = text.split()

    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)

# Sidebar
with st.sidebar:

    st.header("About")

    st.write("""
    This application predicts whether a movie review expresses
    a positive or negative sentiment.

    **Model Used**
    - TF-IDF Vectorizer
    - Logistic Regression
    - IMDB Dataset
    """)

    st.divider()

    st.write("""
    **How to Use**
    
    1. Enter a movie review.
    2. Click Analyze Sentiment.
    3. View prediction and confidence scores.
    """)

# Main Header
st.markdown(
    """
    <div class="big-title">
        🎬 Movie Review Sentiment Analyzer
    </div>

    <div class="subtitle">
        AI-Powered Sentiment Classification using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

# Input Area
review = st.text_area(
    "Enter Your Movie Review",
    height=180,
    placeholder="Example: This movie was absolutely amazing with outstanding acting and storytelling..."
)

# Prediction
if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a movie review.")
    else:

        cleaned = clean_text(review)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        probs = model.predict_proba(vector)[0]

        negative_score = probs[0] * 100
        positive_score = probs[1] * 100

        st.divider()

        if prediction.lower() == "positive":

            st.markdown(
                """
                <div class="result-positive">
                Predicted Sentiment: POSITIVE
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-negative">
                Predicted Sentiment: NEGATIVE
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Negative Confidence",
                f"{negative_score:.2f}%"
            )
            st.progress(int(negative_score))

        with col2:
            st.metric(
                "Positive Confidence",
                f"{positive_score:.2f}%"
            )
            st.progress(int(positive_score))

# Footer
st.divider()

st.caption(
    "Developed by Prabhakar K | NLP Sentiment Analysis Project"
)