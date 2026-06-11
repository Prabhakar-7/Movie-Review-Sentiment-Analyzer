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

# Premium Enterprise SaaS UI Styling
st.markdown("""
<style>
/* Main App Canvas */
.stApp {
    background: radial-gradient(circle at 50% 30%, #1e293b 0%, #0f172a 70%, #020617 100%);
    color: #f8fafc;
}

/* Sidebar Refinements */
[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid #1e293b;
}

/* Force Sidebar Tabs to behave like an app-switcher */
div[data-testid="stTabBar"] {
    background-color: #111827;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #1e293b;
}
button[data-baseweb="tab"] {
    flex: 1;
    text-align: center;
    justify-content: center;
}

/* Centralized Workspace Container Card */
.workspace-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 35px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
}

/* Central Header Typography */
.big-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 30px;
}

/* Label styling refinement */
.field-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #38bdf8;
    margin-bottom: 8px;
}

/* Text Area Refinement */
textarea {
    background-color: #090d16 !important;
    color: #f1f5f9 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 14px !important;
    line-height: 1.6 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}

/* ---- BUTTON FIX BLOCK ---- */
/* This ensures the Streamlit layout wrapper does not squeeze the button container */
div.stButton {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin-top: 15px;
}

/* Targets the core button object securely */
div.stButton > button {
    width: 280px !important;  /* Hardcoded fixed-width so text never wraps or breaks */
    height: 48px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(3, 105, 161, 0.2) !important;
    white-space: nowrap !important; /* Forces the text to stay on a single line */
    transition: all 0.2s ease !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.35) !important;
    border-color: transparent !important;
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(1px);
}
/* -------------------------- */

/* Status Alert Designs */
.result-positive {
    background-color: rgba(16, 185, 129, 0.06);
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 5px solid #10b981;
    color: #34d399;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.03em;
}

.result-negative {
    background-color: rgba(239, 68, 68, 0.06);
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-left: 5px solid #ef4444;
    color: #f87171;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* Metrics Dashboard Cards */
div[data-testid="stMetric"] {
    background-color: #0b0f19;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)

# Load Model and Vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Sidebar Control Console
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=50)
    st.markdown("<h2 style='margin-top:10px;'>Workspace Control</h2>", unsafe_allow_html=True)
    st.caption("Configure environment parameters & metrics.")
    
    st.write("")

    tab1, tab2 = st.tabs(["📊 Engine", "💡 Guide"])
    
    with tab1:
        st.markdown("### Model Stack")
        st.info("**Vectorization:** TF-IDF\n\n**Classifier:** Logistic Regression")
        st.markdown("### Dataset")
        st.caption("Evaluated against the IMDB reference archive (50,000 processed samples).")
        
    with tab2:
        st.markdown("### Framework Execution")
        st.markdown("""
        1. Inject text evaluation string.
        2. Fire inference microservice.
        3. Read confidence arrays.
        """)
        
    st.divider()
    st.markdown("### Quick-Load String Buffers")
    st.caption("Select and paste via clipboard:")
    st.code("An absolute masterpiece with spectacular cinematography.", language="text")
    st.code("Terrible pacing, weak script development, and poor execution.", language="text")

# Main Dashboard Viewport
st.markdown(
    """
    <div class="workspace-card">
        <div class="big-title">Movie Review Sentiment Analyzer</div>
        <div class="subtitle">Enterprise Natural Language Processing Inference Pipeline</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Content Input Section
st.markdown('<div class="field-label">Evaluation Corpus String</div>', unsafe_allow_html=True)
review = st.text_area(
    "Evaluation Corpus String", 
    label_visibility="collapsed",
    height=160,
    placeholder="Stream runtime text strings here for sentiment classification..."
)

# Action Trigger Execution
if st.button("Predict"):
    if not review.strip():
        st.warning("Execution interrupted: Input buffer empty.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)[0]

        negative_score = probs[0] * 100
        positive_score = probs[1] * 100

        st.write("")
        st.divider()

        # Render Professional Cards
        if prediction.lower() == "positive":
            st.markdown(
                """
                <div class="result-positive">
                STATUS: SUCCESS // CLASSIFICATION ENGINE RETURNED: POSITIVE SENTIMENT
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-negative">
                STATUS: SUCCESS // CLASSIFICATION ENGINE RETURNED: NEGATIVE SENTIMENT
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Metrics Panel
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Negative Class Vector Weight", f"{negative_score:.2f}%")
            st.progress(int(negative_score))

        with col2:
            st.metric("Positive Class Vector Weight", f"{positive_score:.2f}%")
            st.progress(int(positive_score))

# Global Framework Footer
st.divider()
st.caption(
    "System Node Status: Online | NLP Analytics Cluster v1.4.2 | Architect: Prabhakar K"
)