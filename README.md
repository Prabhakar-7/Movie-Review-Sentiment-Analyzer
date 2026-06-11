# Movie Review Sentiment Analyzer

## Overview

This project is a Natural Language Processing (NLP) application that classifies movie reviews as either **Positive** or **Negative** using machine learning techniques. The model is trained on the IMDB Movie Reviews Dataset and utilizes TF-IDF feature extraction combined with a Logistic Regression classifier to perform sentiment analysis on unseen reviews.

The application includes a complete machine learning pipeline covering data preprocessing, feature engineering, model training, evaluation, and inference. A Streamlit-based interface can be used to provide real-time sentiment predictions for user-entered reviews.

---

## Features

* Text preprocessing and cleaning
* HTML tag removal
* Tokenization using NLTK
* Stopword removal
* TF-IDF vectorization
* Logistic Regression classifier
* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
* Custom review sentiment prediction
* Model serialization using Joblib
* Streamlit deployment support

---

## Dataset

The project uses the IMDB Movie Reviews Dataset containing 50,000 labeled movie reviews.

Dataset Structure:

| Column    | Description                         |
| --------- | ----------------------------------- |
| review    | Raw movie review text               |
| sentiment | Sentiment label (positive/negative) |

Class Distribution:

* Positive Reviews: 25,000
* Negative Reviews: 25,000

---

## Project Workflow

### 1. Data Loading

The IMDB dataset is loaded into a Pandas DataFrame for processing.

### 2. Text Preprocessing

The preprocessing pipeline performs:

* HTML tag removal
* Removal of punctuation and special characters
* Lowercasing
* Tokenization
* Stopword removal

Example:

Input:

This movie was absolutely amazing! I loved every minute of it.

Processed Output:

movie absolutely amazing loved every minute

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts textual reviews into numerical vectors.

Configuration:

* Maximum Features: 2000

### 4. Model Training

A Logistic Regression classifier is trained on the transformed TF-IDF features.

### 5. Model Evaluation

The trained model is evaluated using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 6. Inference

Users can submit custom movie reviews and receive:

* Predicted sentiment
* Confidence scores for each class

---

## Technology Stack

### Programming Language

* Python 3.x

### Libraries

* Pandas
* NumPy
* NLTK
* Scikit-learn
* Joblib
* Streamlit

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Movie-Review-Sentiment-Analyzer
```

Create and activate a virtual environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the Jupyter Notebook and execute all cells to:

1. Load the dataset
2. Clean the text
3. Generate TF-IDF features
4. Train the classifier
5. Evaluate performance
6. Save model artifacts

Save artifacts:

```python
import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

---

## Running Predictions

Example:

```python
review = "This movie was fantastic and highly entertaining."

cleaned_review = clean_text(review)

vector = tfidf.transform([cleaned_review])

prediction = model.predict(vector)

print(prediction)
```

---

## Model Evaluation Metrics

The project reports:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 89.5% |
| Precision | 89.0% |
| Recall    | 90.0% |
| F1 Score  | 89.5% |

*Actual values may vary depending on train-test split and preprocessing configuration.*

---

## Project Structure

```text
Movie-Review-Sentiment-Analyzer/

├── notebook.ipynb
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
├── IMDB Dataset.csv
└── .gitignore
```

---

## Streamlit Deployment

Run the application locally:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Future Improvements

Potential enhancements include:

* Lemmatization using spaCy
* Word2Vec embeddings
* GloVe embeddings
* Support Vector Machine (SVM) classifier
* Hyperparameter tuning
* Cross-validation
* Deep learning models (LSTM, GRU)
* Transformer-based models (BERT, RoBERTa)
* REST API deployment using FastAPI
* Docker containerization
* CI/CD integration

---

## License

This project is intended for educational and research purposes.

---

## Author

Prabhakar K

Machine Learning and Natural Language Processing Enthusiast
