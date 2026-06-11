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

# Premium Professional Cinema/Enterprise UI Styling
st.markdown("""
<style>
/* Main App Window Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #0b0f19 100%);
    color: #f1f5f9;
}

/* Sidebar Styling: Modern Frosted Glass Look */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1f2937;
}

/* Form inputs & Textarea refinement */
textarea {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    transition: all 0.3s ease;
}
textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* Title: Clean, High-Contrast Premium Typography */
.big-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 500;
    margin-bottom: 35px;
}

/* Professional Result Banner: Positive */
.result-positive {
    background-color: rgba(16, 185, 129, 0.1);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-left: 6px solid #10b981;
    color: #34d399;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Professional Result Banner: Negative */
.result-negative {
    background-color: rgba(239, 68, 68, 0.1);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 6px solid #ef4444;
    color: #f87171;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Master Primary Action Button */
.stButton button {
    width: 100%;
    height: 3.2em;
    font-size: 15px;
    font-weight: 600;
    background-color: #2563eb;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}

.stButton button:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
    transform: translateY(-1px);
}

.stButton button:active {
    transform: translateY(0px);
}

/* Metric Display Panels */
div[data-testid="stMetric"] {
    background-color: #131c2e;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #1e293b;
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
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Interactive Workspace Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=60)
    st.title("Workspace Control")
    st.write("Configure and inspect engine settings.")
    
    st.space() # Clean spacing

    # Interactive Component: Tabbed Layout for cleaner presentation
    tab1, tab2 = st.tabs(["📊 Model Details", "💡 User Guide"])
    
    with tab1:
        st.markdown("### Architecture Stack")
        st.info("**Vectorization:** TF-IDF\n\n**Classifier:** Logistic Regression")
        st.markdown("### Training Data Source")
        st.caption("Trained on the highly benchmarked IMDB Movie Reviews Dataset containing 50,000 highly polar reviews.")
        
    with tab2:
        st.markdown("### Operation Steps")
        st.markdown("""
        1. **Input text** into the main viewport area.
        2. Execute the inference engine by pressing **Analyze Sentiment**.
        3. Real-time class probabilities will render instantly below.
        """)
        
    st.divider()
    # Dynamic component to interactively try quick samples
    st.markdown("### Test Vectors Quick-Load")
    st.caption("Copy-paste these clean test cases if you need quick evaluations:")
    st.code("An absolute cinematic masterpiece with breathtaking execution.", language="text")
    st.code("A complete disaster. Waste of time, shallow characters, and poor script.", language="text")

# Main Header Area
st.markdown(
    """
    <div class="big-title">
        Movie Review Sentiment Analyzer
    </div>
    <div class="subtitle">
        Enterprise Natural Language Processing Inference Pipeline
    </div>
    """,
    unsafe_allow_html=True
)

# Input viewport Area
review = st.text_area(
    "Evaluation Corpus String",
    height=180,
    placeholder="Analyze runtime strings here (e.g., 'The structural pacing of the film fell flat during act two...')"
)

# Inference Execution Block
if st.button("Execute Inference Pipeline"):
    if not review.strip():
        st.warning("Execution halted: Input corpus cannot be null or blank.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        negative_score = probs[0] * 100
        positive_score = probs[1] * 100

        st.divider()

        # Render Professional Alert Cards
        if prediction.lower() == "positive":
            st.markdown(
                """
                <div class="result-positive">
                ANALYSIS COMPLETE // CLASSIFICATION: POSITIVE SENTIMENT
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-negative">
                ANALYSIS COMPLETE // CLASSIFICATION: NEGATIVE SENTIMENT
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Probability Metric Cards
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Negative Class Probability",
                f"{negative_score:.2f}%"
            )
            st.progress(int(negative_score))

        with col2:
            st.metric(
                "Positive Class Probability",
                f"{positive_score:.2f}%"
            )
            st.progress(int(positive_score))

# Global Footer
st.divider()
st.caption(
    "System Status: Operational | NLP Processing Engine Node | Architecture by Prabhakar K"
)